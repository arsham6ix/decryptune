// Rewrite a fragmented MP4 (init segment + moof/mdat pairs) as a classic MP4:
// the moov's real sample tables (stts/stsc/stsz/stco) are rebuilt from the
// fragments, mvex/moof/sidx disappear, and all sample bytes land in one mdat.
// Audio data is byte-copied — never re-encoded. Scope: single audio track,
// which is the entire decryptune surface. Ported from a trusted HLS
// postprocessor and hardened for hostile input: every field read is bounded
// by its own box, sample data must lie inside the file, and the output is
// checked against the 32-bit chunk-offset limit before anything is written.

use crate::boxes::{header_len, parse, write_header, Box, MAX_BOXES};
use crate::process::trex_defaults;
use crate::Err;

// containers the moov rebuild recurses into (mvex is dropped outright)
const CONTAINERS: [&[u8; 4]; 6] = [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"];
const MAX_SAMPLES: usize = 1 << 22;

// one contiguous sample-data run referenced by a trun — one output chunk
struct ChunkRun {
    offset : usize, // byte position inside the source buffer
    len    : usize,
    samples: usize
}

// rebuilt sample tables, in stbl append order
struct Tables {
    stts: Vec<u8>,
    stsc: Vec<u8>,
    stsz: Vec<u8>,
    stco: Vec<u8>
}

fn be32(data: &[u8], pos: usize, end: usize) -> Result<u32, Err> {
    if pos + 4 > end { return Err(Err::ClassicUnsupported); }
    Ok(u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()))
}

fn timescale(payload: &[u8]) -> Result<u32, Err> {
    // mvhd/mdhd share the prefix: ver/flags(4) creation(v0:4/v1:8)
    // modification(v0:4/v1:8) timescale(4)
    let version: u8 = *payload.first().ok_or(Err::ClassicUnsupported)?;
    let off: usize = if version == 1 { 20 } else { 12 };
    if off + 4 > payload.len() { return Err(Err::ClassicUnsupported); }
    Ok(u32::from_be_bytes(payload[off..off + 4].try_into().unwrap()))
}

// rewrite the duration field of mvhd/mdhd (16/24) or tkhd (20/28); v0 values
// clamp to u32::MAX like every writer does — audio never gets near it
fn patch_duration(data: &[u8], b: &Box, off_v0: usize, off_v1: usize, value: u64) -> Result<Vec<u8>, Err> {
    let payload: &[u8] = &data[b.content_start..b.end];
    let version: u8 = *payload.first().ok_or(Err::ClassicUnsupported)?;
    let (off, wide): (usize, bool) = if version == 1 { (off_v1, true) } else { (off_v0, false) };
    let need: usize = off + if wide { 8 } else { 4 };
    if need > payload.len() { return Err(Err::ClassicUnsupported); }
    let mut out: Vec<u8> = payload.to_vec();
    if wide {
        out[off..off + 8].copy_from_slice(&value.to_be_bytes());
    } else {
        out[off..off + 4].copy_from_slice(&(value.min(u64::from(u32::MAX)) as u32).to_be_bytes());
    }
    Ok(out)
}

// fill a zero segment duration (normal for fragmented files) in a one-entry
// edit list — a zero elst duration makes players clip the track to nothing
fn fix_elst(data: &[u8], b: &Box, movie_duration: u64) -> Result<Vec<u8>, Err> {
    let payload: &[u8] = &data[b.content_start..b.end];
    if payload.len() < 8 { return Err(Err::ClassicUnsupported); }
    let mut out: Vec<u8> = payload.to_vec();
    let version: u8 = out[0];
    let count: u32 = u32::from_be_bytes(out[4..8].try_into().unwrap());
    if count != 1 { return Ok(out); }
    if version == 1 {
        if out.len() >= 16 && out[8..16] == [0u8; 8] {
            out[8..16].copy_from_slice(&movie_duration.to_be_bytes());
        }
    } else if out.len() >= 12 && out[8..12] == [0u8; 4] {
        out[8..12].copy_from_slice(&(movie_duration.min(u64::from(u32::MAX)) as u32).to_be_bytes());
    }
    Ok(out)
}

fn make_tables(chunks: &[ChunkRun], offsets: &[u32], durations: &[u32], sizes: &[u32]) -> Tables {
    // stts: run-length encode the per-sample durations
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for d in durations {
        match runs.last_mut() {
            Some(r) if r.1 == *d => r.0 += 1,
            _                    => runs.push((1, *d))
        }
    }
    let mut stts: Vec<u8> = Vec::with_capacity(8 + runs.len() * 8);
    stts.extend_from_slice(&0u32.to_be_bytes());
    stts.extend_from_slice(&(runs.len() as u32).to_be_bytes());
    for (n, d) in &runs {
        stts.extend_from_slice(&n.to_be_bytes());
        stts.extend_from_slice(&d.to_be_bytes());
    }

    // stsc: a new entry only where the per-chunk sample count changes
    let mut stsc_entries: Vec<(u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk)
    for (i, c) in chunks.iter().enumerate() {
        let n: u32 = (i + 1) as u32;
        if stsc_entries.last().map(|e| e.1) != Some(c.samples as u32) {
            stsc_entries.push((n, c.samples as u32));
        }
    }
    let mut stsc: Vec<u8> = Vec::with_capacity(8 + stsc_entries.len() * 12);
    stsc.extend_from_slice(&0u32.to_be_bytes());
    stsc.extend_from_slice(&(stsc_entries.len() as u32).to_be_bytes());
    for (first, count) in &stsc_entries {
        stsc.extend_from_slice(&first.to_be_bytes());
        stsc.extend_from_slice(&count.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
    }

    // stsz: a uniform table is 12 bytes instead of one entry per sample
    let uniform: bool = sizes.iter().all(|s| *s == sizes[0]);
    let mut stsz: Vec<u8> = Vec::with_capacity(12 + if uniform { 0 } else { sizes.len() * 4 });
    stsz.extend_from_slice(&0u32.to_be_bytes());
    stsz.extend_from_slice(&if uniform { sizes[0] } else { 0 }.to_be_bytes());
    stsz.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
    if !uniform {
        for s in sizes { stsz.extend_from_slice(&s.to_be_bytes()); }
    }

    let mut stco: Vec<u8> = Vec::with_capacity(8 + offsets.len() * 4);
    stco.extend_from_slice(&0u32.to_be_bytes());
    stco.extend_from_slice(&(offsets.len() as u32).to_be_bytes());
    for o in offsets { stco.extend_from_slice(&o.to_be_bytes()); }

    Tables { stts, stsc, stsz, stco }
}

// recursive moov rebuild — the only structural change is that mvex vanishes
// and the sample tables inside stbl are appended fresh
fn rebuild(data: &[u8], start: usize, end: usize, tables: &Tables, movie_duration: u64, media_duration: u64) -> Result<Vec<u8>, Err> {
    let children: Vec<Box> = parse(data, start, end);
    if children.len() >= MAX_BOXES { return Err(Err::ClassicUnsupported); }
    let mut out: Vec<u8> = Vec::new();
    for b in &children {
        match &b.typ {
            b"mvex" | b"stts" | b"stsc" | b"stsz" | b"stco" | b"co64" | b"stz2" => {} // dropped — fragmentation markers or replaced tables
            b"mvhd" => {
                let body: Vec<u8> = patch_duration(data, b, 16, 24, movie_duration)?;
                write_header(&mut out, b"mvhd", body.len());
                out.extend_from_slice(&body);
            }
            b"mdhd" => {
                let body: Vec<u8> = patch_duration(data, b, 16, 24, media_duration)?;
                write_header(&mut out, b"mdhd", body.len());
                out.extend_from_slice(&body);
            }
            b"tkhd" => {
                let body: Vec<u8> = patch_duration(data, b, 20, 28, movie_duration)?;
                write_header(&mut out, b"tkhd", body.len());
                out.extend_from_slice(&body);
            }
            b"elst" => {
                let body: Vec<u8> = fix_elst(data, b, movie_duration)?;
                write_header(&mut out, b"elst", body.len());
                out.extend_from_slice(&body);
            }
            b"stbl" => {
                let mut inner: Vec<u8> = rebuild(data, b.content_start, b.end, tables, movie_duration, media_duration)?;
                // the Tables fields hold box BODIES — wrap each in its header
                write_header(&mut inner, b"stts", tables.stts.len());
                inner.extend_from_slice(&tables.stts);
                write_header(&mut inner, b"stsc", tables.stsc.len());
                inner.extend_from_slice(&tables.stsc);
                write_header(&mut inner, b"stsz", tables.stsz.len());
                inner.extend_from_slice(&tables.stsz);
                write_header(&mut inner, b"stco", tables.stco.len());
                inner.extend_from_slice(&tables.stco);
                write_header(&mut out, b"stbl", inner.len());
                out.extend_from_slice(&inner);
            }
            typ if CONTAINERS.iter().any(|c| c == &typ) => {
                let inner: Vec<u8> = rebuild(data, b.content_start, b.end, tables, movie_duration, media_duration)?;
                write_header(&mut out, typ, inner.len());
                out.extend_from_slice(&inner);
            }
            _ => out.extend_from_slice(&data[b.start..b.end])
        }
    }
    Ok(out)
}

/// Rewrite fragmented MP4 bytes as a classic MP4. A file without moof boxes
/// is already classic and returns unchanged.
pub(crate) fn defragment(data: &[u8]) -> Result<Vec<u8>, Err> {
    let top: Vec<Box> = parse(data, 0, data.len());
    if top.is_empty() || top.len() >= MAX_BOXES { return Err(Err::ClassicUnsupported); }
    if !top.iter().any(|b| b.is(b"moof")) { return Ok(data.to_vec()); }
    let moov: &Box = top.iter().find(|b| b.is(b"moov")).ok_or(Err::ClassicUnsupported)?;
    let children: Vec<Box> = parse(data, moov.content_start, moov.end);
    if children.len() >= MAX_BOXES || children.iter().filter(|b| b.is(b"trak")).count() != 1 {
        return Err(Err::ClassicUnsupported);
    }
    let mvhd: &Box = children.iter().find(|b| b.is(b"mvhd")).ok_or(Err::ClassicUnsupported)?;
    let trak: &Box = children.iter().find(|b| b.is(b"trak")).ok_or(Err::ClassicUnsupported)?;
    let trak_children: Vec<Box> = parse(data, trak.content_start, trak.end);
    let mdia: &Box = trak_children.iter().find(|b| b.is(b"mdia")).ok_or(Err::ClassicUnsupported)?;
    let mdia_children: Vec<Box> = parse(data, mdia.content_start, mdia.end);
    let mdhd: &Box = mdia_children.iter().find(|b| b.is(b"mdhd")).ok_or(Err::ClassicUnsupported)?;
    let movie_ts: u32 = timescale(&data[mvhd.content_start..mvhd.end])?;
    let media_ts: u32 = timescale(&data[mdhd.content_start..mdhd.end])?;
    if movie_ts == 0 || media_ts == 0 { return Err(Err::ClassicUnsupported); }

    // walk the fragments: sample sizes, durations, and where the bytes are.
    // base_data_offset defaults to the moof start; sample defaults cascade
    // trex → tfhd → trun entry, exactly like the decrypt planner
    let trex: Option<(u32, u32)> = trex_defaults(data, &top);
    let (mut sizes, mut durations): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut chunks: Vec<ChunkRun> = Vec::new();
    for moof in top.iter().filter(|b| b.is(b"moof")) {
        let moof_children: Vec<Box> = parse(data, moof.content_start, moof.end);
        for traf in &moof_children {
            if !traf.is(b"traf") { continue; }
            let subs: Vec<Box> = parse(data, traf.content_start, traf.end);
            let mut base: usize = moof.start;
            let mut dur: u32 = trex.map(|(d, _)| d).unwrap_or(0);
            let mut size: u32 = trex.map(|(_, s)| s).unwrap_or(0);
            let mut next: Option<usize> = None;
            for sub in &subs {
                match &sub.typ {
                    b"tfhd" => {
                        let flags: u32 = be32(data, sub.content_start, sub.end)? & 0xFFFFFF;
                        if sub.content_start + 8 > sub.end { return Err(Err::ClassicUnsupported); }
                        let mut q: usize = sub.content_start + 8; // version/flags + track_ID
                        if flags & 0x1 != 0 {  // base_data_offset (absolute)
                            if q + 8 > sub.end { return Err(Err::ClassicUnsupported); }
                            let b64: u64 = u64::from_be_bytes(data[q..q + 8].try_into().unwrap());
                            if b64 > data.len() as u64 { return Err(Err::ClassicUnsupported); }
                            base = b64 as usize;
                            q += 8;
                        }
                        if flags & 0x2 != 0 { q += 4; }  // sample_description_index — unused
                        if flags & 0x8 != 0 { dur = be32(data, q, sub.end)?; q += 4; }
                        if flags & 0x10 != 0 { size = be32(data, q, sub.end)?; q += 4; }
                    }
                    b"trun" => {
                        let flags: u32 = be32(data, sub.content_start, sub.end)? & 0xFFFFFF;
                        let count: usize = be32(data, sub.content_start + 4, sub.end)? as usize;
                        if count > MAX_SAMPLES { return Err(Err::ClassicUnsupported); }
                        let mut q: usize = sub.content_start + 8;
                        let data_start: usize;
                        if flags & 0x1 != 0 {  // data_offset, signed, relative to base
                            let off: i32 = be32(data, q, sub.end)? as i32;
                            q += 4;
                            let abs: i64 = base as i64 + off as i64;
                            if abs < 0 || abs > data.len() as i64 { return Err(Err::ClassicUnsupported); }
                            data_start = abs as usize;
                        } else if let Some(n) = next {
                            data_start = n;  // chained after the previous run in this moof
                        } else {
                            return Err(Err::ClassicUnsupported);
                        }
                        if flags & 0x4 != 0 { q += 4; }  // first_sample_flags — unused
                        let mut run_len: usize = 0;
                        for _ in 0..count {
                            let mut d: u32 = dur;
                            let mut s: u32 = size;
                            if flags & 0x100 != 0 { d = be32(data, q, sub.end)?; q += 4; }
                            if flags & 0x200 != 0 { s = be32(data, q, sub.end)?; q += 4; }
                            if flags & 0x400 != 0 { q += 4; }  // sample_flags — unused
                            if flags & 0x800 != 0 { q += 4; }  // composition offsets — unused
                            if s == 0 { return Err(Err::ClassicUnsupported); } // a sample without bytes cannot be placed
                            sizes.push(s);
                            durations.push(d);
                            run_len += s as usize;
                        }
                        if data_start + run_len > data.len() { return Err(Err::ClassicUnsupported); }
                        if count > 0 {
                            chunks.push(ChunkRun { offset: data_start, len: run_len, samples: count });
                            next = Some(data_start + run_len);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if sizes.is_empty() { return Err(Err::ClassicUnsupported); }
    let media_duration: u64 = durations.iter().map(|d| u64::from(*d)).sum();
    if media_duration == 0 { return Err(Err::ClassicUnsupported); }
    let movie_duration: u64 = media_duration * u64::from(movie_ts) / u64::from(media_ts);

    // moov's size does not depend on the offset VALUES — size it with dummy
    // offsets first, then place the data and rebuild with the real ones
    let zeros: Vec<u32> = vec![0u32; chunks.len()];
    let dummy: Tables = make_tables(&chunks, &zeros, &durations, &sizes);
    let placeholder: Vec<u8> = rebuild(data, moov.content_start, moov.end, &dummy, movie_duration, media_duration)?;
    let others_len: usize = top.iter()
        .filter(|b| !matches!(&b.typ, b"moof" | b"mdat" | b"sidx" | b"moov"))
        .map(|b| b.end - b.start)
        .sum();
    let data_start: usize = others_len + header_len(placeholder.len()) + placeholder.len() + 8; // moov box + mdat header
    let total_audio: usize = chunks.iter().map(|c| c.len).sum();
    if data_start + total_audio > u32::MAX as usize {
        return Err(Err::ClassicUnsupported); // stco holds u32 offsets only (audio never gets there)
    }
    let mut offsets: Vec<u32> = Vec::with_capacity(chunks.len());
    let mut pos: usize = data_start;
    for c in &chunks {
        offsets.push(pos as u32);
        pos += c.len;
    }
    let tables: Tables = make_tables(&chunks, &offsets, &durations, &sizes);
    let new_moov: Vec<u8> = rebuild(data, moov.content_start, moov.end, &tables, movie_duration, media_duration)?;
    if new_moov.len() != placeholder.len() { return Err(Err::ClassicUnsupported); } // cannot happen — table sizes are value-independent

    // unknown top-level boxes (free, top-level meta…) are kept verbatim;
    // moof/mdat/sidx fold into the tables and the single output mdat
    let mut out: Vec<u8> = Vec::with_capacity(data_start + total_audio);
    for b in &top {
        match &b.typ {
            b"moof" | b"mdat" | b"sidx" => {}
            b"moov" => {
                write_header(&mut out, b"moov", new_moov.len());
                out.extend_from_slice(&new_moov);
            }
            _       => out.extend_from_slice(&data[b.start..b.end])
        }
    }
    write_header(&mut out, b"mdat", total_audio);
    for c in &chunks {
        out.extend_from_slice(&data[c.offset..c.offset + c.len]);
    }
    Ok(out)
}
