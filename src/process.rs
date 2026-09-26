use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};

use crate::boxes::{parse, write_header, Box};
use crate::meta::{rebuild_moov_with_meta, CoreMeta};
use crate::Err;

type Aes128Ctr = ctr::Ctr128BE<Aes128>;

const CONTAINERS: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl"];
const MAX_TREE_DEPTH: usize = 16;
// chi2 thresholds calibrated on real AAC tracks (structured ~3800..14000, noise ~210..340)
const MIN_JUDGE_BYTES: usize = 16 * 1024; // below this, chi2 drowns in the df=255 noise floor
const CHI2_CORRUPT_BELOW: f64 = 500.0;

#[allow(dead_code)] // fields beyond `corrupt` are diagnostics for future reporting
pub(crate) struct FragmentStat {
    pub index     : usize,
    pub encrypted : bool,
    pub bytes     : usize,
    pub chi2      : f64,
    pub corrupt   : Option<bool> // Some(true)=corrupt, Some(false)=clean, None=not judged
}

pub(crate) struct Processed {
    pub data      : Vec<u8>,
    pub fragments : Vec<FragmentStat>
}

struct Plan {
    ivs         : Option<Vec<[u8; 16]>>,
    sizes       : Vec<u32>,
    default     : Option<u32>,
    ok          : bool,
    rebuilt_len : usize
}

// Fused single-walk: decrypt + fix + metadata in one pass, one output buffer
// (key = None → fix-only pass)
pub fn process(data: &[u8], key: Option<&[u8; 16]>, strict: bool, meta: Option<&CoreMeta>) -> Result<Processed, Err> {
    let boxes: Vec<Box> = parse(data, 0, data.len());
    if boxes.is_empty() { return Err(Err::InvalidMp4); }

    let mut out: Vec<u8> = Vec::with_capacity(data.len());
    let mut deltas: Vec<i64> = Vec::new();
    let mut patches: Vec<(usize, Vec<usize>)> = Vec::new(); // (frag base, sidx word offsets)
    let mut fragments: Vec<FragmentStat> = Vec::new();

    let mut i = 0;
    while i < boxes.len() {
        let b: &Box = &boxes[i];
        match &b.typ {
            b"moof" => {
                let plan: Plan = visit_moof(&mut out, data, b);
                deltas.push((b.end - b.start) as i64 - plan.rebuilt_len as i64);
                if i + 1 < boxes.len() && boxes[i + 1].is(b"mdat") {
                    let stat: FragmentStat = emit_mdat(&mut out, data, &boxes[i + 1], key, &plan, strict);
                    fragments.push(FragmentStat { index: fragments.len(), ..stat });
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }
            b"moov" => {
                let children: Vec<u8> = rebuild_container(data, b.content_start, b.end, 0);
                let children: Vec<u8> = if let Some(m) = meta {
                    // growing moov dangles absolute chunk offsets — fragmented files carry empty stco
                    if has_chunk_offsets(&children, 0) { return Err(Err::MetaNeedsFragmented); }
                    rebuild_moov_with_meta(&children, m)?
                } else {
                    children
                };
                write_header(&mut out, b"moov", children.len());
                out.extend_from_slice(&children);
            }
            b"sidx" => {
                let base: usize = out.len();
                out.extend_from_slice(&data[b.start..b.end]);
                patches.push((deltas.len(), sidx_words(data, b, base)));
            }
            _ => out.extend_from_slice(&data[b.start..b.end])
        }
        i += 1;
    }

    // patch sidx referenced sizes in place, now that fragment deltas are known
    for (frag_base, words) in &patches {
        for (k, &off) in words.iter().enumerate() {
        let fi: usize = frag_base + k;
        if fi >= deltas.len() || deltas[fi] <= 0 || off + 4 > out.len() { continue; }
        let d: u32 = (deltas[fi] as u64).min(0x7FFF_FFFF) as u32;
        let word: u32 = u32::from_be_bytes(out[off..off + 4].try_into().unwrap());
        let new: u32 = (word & 0x7FFF_FFFF).saturating_sub(d);
            out[off..off + 4].copy_from_slice(&((word & 0x8000_0000) | (new & 0x7FFF_FFFF)).to_be_bytes());
        }
    }

    Ok(Processed { data: out, fragments })
}

fn visit_moof(out: &mut Vec<u8>, data: &[u8], b: &Box) -> Plan {
    let mut content: Vec<u8> = Vec::new();
    let mut ivs: Option<Vec<[u8; 16]>> = None;
    let mut sizes: Vec<u32> = Vec::new();
    let mut default_size: Option<u32> = None;

    for child in parse(data, b.content_start, b.end) {
        if !child.is(b"traf") {
            content.extend_from_slice(&data[child.start..child.end]);
            continue;
        }
        let traf: Vec<Box> = parse(data, child.content_start, child.end);

        // gather: tfhd default first so trun fallback works in any order
        default_size = traf.iter().find(|t| t.is(b"tfhd"))
            .and_then(|t| parse_tfhd(data, t.content_start));
        for t in &traf {
            match &t.typ {
                b"senc" if ivs.is_none() => ivs = parse_senc(data, t.content_start),
                b"trun"                  => sizes.extend(parse_trun_sizes(data, t.content_start, t.end, default_size)),
                _                        => {}
            }
        }

        // rebuild: drop encryption boxes, shrink trun data offsets
        let removed: i64 = traf.iter()
            .filter(|t| matches!(&t.typ, b"senc" | b"saiz" | b"saio"))
            .map(|t| (t.end - t.start) as i64)
            .sum();
        let mut traf_out: Vec<u8> = Vec::new();
        for t in &traf {
            match &t.typ {
                b"senc" | b"saiz" | b"saio" => {}
                b"trun"                     => {
                    write_header(&mut traf_out, b"trun", t.end - t.content_start);
                    traf_out.extend_from_slice(&patch_trun(data, t.content_start, t.end, removed));
                }
                _ => traf_out.extend_from_slice(&data[t.start..t.end])
            }
        }
        write_header(&mut content, b"traf", traf_out.len());
        content.extend_from_slice(&traf_out);
    }

    // senc/trun count mismatch → partially plain fragment (part of the track is
    // stored in the clear); skip decryption for it
    let ok: bool = matches!(&ivs, Some(list) if !sizes.is_empty() && list.len() == sizes.len());

    write_header(out, b"moof", content.len());
    out.extend_from_slice(&content);
    Plan { ivs, sizes, default: default_size, ok, rebuilt_len: content.len() + 8 }
}

fn chi2(buf: &[u8]) -> f64 {
    let n: usize = buf.len();
    if n == 0 { return 0.0; }
    let mut hist: [u64; 256] = [0u64; 256];
    for &b in buf { hist[b as usize] += 1; }
    let e: f64 = n as f64 / 256.0;
    hist.iter().map(|&o| { let d: f64 = o as f64 - e; d * d / e }).sum()
}

fn emit_mdat(
    out    : &mut Vec<u8>,
    data   : &[u8],
    m      : &Box,
    key    : Option<&[u8; 16]>,
    plan   : &Plan,
    strict : bool
) -> FragmentStat {
    write_header(out, b"mdat", m.end - m.content_start);
    let base: usize = out.len();
    out.extend_from_slice(&data[m.content_start..m.end]);
    let encrypted: bool = plan.ok;
    if encrypted {
        if let (Some(iv_list), Some(key)) = (&plan.ivs, key) {
            let mut pos: usize = base;
            for (iv, &sz) in iv_list.iter().zip(plan.sizes.iter()) {
                // a zero entry in a per-entry table means "use the tfhd default"
                let sz: usize = if sz == 0 { plan.default.unwrap_or(0) } else { sz } as usize;
                if sz == 0 || pos + sz > out.len() { break; } // continuing would misalign every later sample
                Aes128Ctr::new(key.into(), iv.into()).apply_keystream(&mut out[pos..pos + sz]);
                pos += sz;
            }
        }
    }

    let payload: &[u8] = &out[base..];
    // chi2 costs a full pass over the payload — pay it only when a verdict is requested
    let judged: bool = encrypted && key.is_some() && strict && payload.len() >= MIN_JUDGE_BYTES;
    let chi2: f64 = if judged { chi2(payload) } else { 0.0 };
    let corrupt = if !encrypted { Some(false) } else if !judged { None } else { Some(chi2 < CHI2_CORRUPT_BELOW) };
    FragmentStat { index: 0, encrypted, bytes: payload.len(), chi2, corrupt }
}

// absolute chunk offsets dangle once moov grows — fragmented files carry empty stco
fn has_chunk_offsets(children: &[u8], depth: usize) -> bool {
    if depth > MAX_TREE_DEPTH { return false; }
    for child in parse(children, 0, children.len()) {
        match &child.typ {
            b"stco" if child.end - child.content_start >= 8
                && u32::from_be_bytes(children[child.content_start + 4..child.content_start + 8].try_into().unwrap()) > 0 => return true,
            b"co64" if child.end - child.content_start >= 12
                && u64::from_be_bytes(children[child.content_start + 4..child.content_start + 12].try_into().unwrap()) > 0 => return true,
            typ if matches!(typ, b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl") => {
                if has_chunk_offsets(&children[child.content_start..child.end], depth + 1) { return true; }
            }
            _ => {}
        }
    }
    false
}

// default_sample_size from tfhd (flag 0x10), honoring preceding optional fields
fn parse_tfhd(data: &[u8], start: usize) -> Option<u32> {
    if start + 8 > data.len() { return None; }
    let flags: u32 = u32::from_be_bytes(data[start..start + 4].try_into().ok()?) & 0xFFFFFF;
    let mut pos: usize = start + 8; // version/flags + track_ID
    if flags & 0x1 != 0 { pos += 8; }  // base_data_offset
    if flags & 0x2 != 0 { pos += 4; }  // sample_description_index
    if flags & 0x8 != 0 { pos += 4; }  // default_sample_duration
    if flags & 0x10 == 0 || pos + 4 > data.len() { return None; }
    Some(u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?))
}

// IVs from senc; subsample entries (flag 0x2) are parsed over for IV alignment,
// not applied — samples are fully encrypted
fn parse_senc(data: &[u8], start: usize) -> Option<Vec<[u8; 16]>> {
    if start + 8 > data.len() { return None; }
    let flags: u32 = u32::from_be_bytes(data[start..start + 4].try_into().ok()?) & 0xFFFFFF;
    let count: usize = u32::from_be_bytes(data[start + 4..start + 8].try_into().ok()?) as usize;
    if count > (data.len() - start - 8) / 8 { return None; } // hostile count bound
    let mut pos: usize = start + 8;
    let mut ivs: Vec<[u8; 16]> = Vec::with_capacity(count);
    for _ in 0..count {
        if pos + 8 > data.len() { return None; }
        let mut iv: [u8; 16] = [0u8; 16];
        iv[..8].copy_from_slice(&data[pos..pos + 8]);
        pos += 8;
        if flags & 0x2 != 0 {
            if pos + 2 > data.len() { return None; }
            let sub: usize = u16::from_be_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2 + sub * 6;
        }
        ivs.push(iv);
    }
    Some(ivs)
}

fn parse_trun_sizes(data: &[u8], start: usize, end: usize, default: Option<u32>) -> Vec<u32> {
    if start + 8 > end { return Vec::new(); }
    let flags: u32 = u32::from_be_bytes(data[start..start + 4].try_into().unwrap()) & 0xFFFFFF;
    let count: usize = u32::from_be_bytes(data[start + 4..start + 8].try_into().unwrap()) as usize;
    let mut pos: usize = start + 8;
    if flags & 0x1 != 0 { pos += 4; }  // data_offset
    if flags & 0x4 != 0 { pos += 4; }  // first_sample_flags
    let stride: usize =
        (if flags & 0x100 != 0 { 4 } else { 0 }) +
        (if flags & 0x200 != 0 { 4 } else { 0 }) +
        (if flags & 0x400 != 0 { 4 } else { 0 }) +
        (if flags & 0x800 != 0 { 4 } else { 0 });
    let size_off: usize = if flags & 0x100 != 0 { 4usize } else { 0 };
    // hostile count bound: an entry table cannot exceed the box; a default
    // fill cannot exceed a quarter of the file (one u32 size each)
    let count: usize = if flags & 0x200 != 0 && stride != 0 {
        count.min(end.saturating_sub(pos) / stride) // pos may sit past end on hostile flags
    } else {
        count.min(data.len() / 4)
    };
    if flags & 0x200 == 0 || stride == 0 {
        return vec![default.unwrap_or(0); count];
    }
    (0..count).map(|i| {
        let base: usize = pos + i * stride + size_off;
        if base + 4 > end { return 0; }
        u32::from_be_bytes(data[base..base + 4].try_into().unwrap())
    }).collect()
}

fn patch_trun(data: &[u8], start: usize, end: usize, removed: i64) -> Vec<u8> {
    let mut out: Vec<u8> = data[start..end].to_vec();
    if out.len() >= 12 {
        let flags: u32 = u32::from_be_bytes(out[0..4].try_into().unwrap()) & 0xFFFFFF;
        if flags & 0x1 != 0 {
            let orig: i64 = i32::from_be_bytes(out[8..12].try_into().unwrap()) as i64;
            out[8..12].copy_from_slice(&((orig - removed) as i32).to_be_bytes());
        }
    }
    out
}

// output offsets of each sidx referenced_size word, patched in place later
fn sidx_words(data: &[u8], b: &Box, out_base: usize) -> Vec<usize> {
    let start: usize = b.content_start;
    if start + 4 > b.end { return Vec::new(); }
    let time_len: usize = if data[start] == 0 { 8usize } else { 16 }; // covers EPT + first_offset together
    let entries: usize = 4 + 4 + 4 + time_len + 2 + 2;
    if start + entries > b.end { return Vec::new(); } // count word — the v1 header is longer than v0
    let count: usize = u16::from_be_bytes(data[start + entries - 2..start + entries].try_into().unwrap()) as usize;
    if start + entries + count * 12 > b.end { return Vec::new(); }
    (0..count).map(|k| out_base + (start - b.start) + entries + k * 12).collect()
}

fn rebuild_container(data: &[u8], start: usize, end: usize, depth: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for child in parse(data, start, end) {
        match &child.typ {
            b"pssh" => {}
            b"stsd" => {
                let content: Vec<u8> = rebuild_stsd(data, child.content_start, child.end);
                write_header(&mut out, b"stsd", content.len());
                out.extend_from_slice(&content);
            }
            typ if CONTAINERS.contains(&typ) && depth < MAX_TREE_DEPTH => {
                let content: Vec<u8> = rebuild_container(data, child.content_start, child.end, depth + 1);
                write_header(&mut out, typ, content.len());
                out.extend_from_slice(&content);
            }
            _ => out.extend_from_slice(&data[child.start..child.end])
        }
    }
    out
}

fn rebuild_stsd(data: &[u8], start: usize, end: usize) -> Vec<u8> {
    if end - start < 8 { return data[start..end].to_vec(); }
    let mut out: Vec<u8> = data[start..start + 8].to_vec();
    for entry in parse(data, start + 8, end) {
        if entry.is(b"enca") {
            // the plaintext fourcc lives in sinf/frma — never hardcode mp4a
            let (content, fourcc): (Vec<u8>, [u8; 4]) = rebuild_enca(data, entry.content_start, entry.end);
            write_header(&mut out, &fourcc, content.len());
            out.extend_from_slice(&content);
        } else if !entry.is(b"sinf") {
            // sinf belongs inside the sample entry; drop stray siblings too
            out.extend_from_slice(&data[entry.start..entry.end]);
        }
    }
    out
}

fn rebuild_enca(data: &[u8], start: usize, end: usize) -> (Vec<u8>, [u8; 4]) {
    let mut fourcc: [u8; 4] = *b"mp4a";
    if end - start < 28 { return (data[start..end].to_vec(), fourcc); }
    let mut out: Vec<u8> = data[start..start + 28].to_vec();
    for child in parse(data, start + 28, end) {
        if child.is(b"sinf") {
            for sub in parse(data, child.content_start, child.end) {
                if sub.is(b"frma") && sub.end - sub.content_start >= 4 {
                    fourcc.copy_from_slice(&data[sub.content_start..sub.content_start + 4]);
                }
            }
        } else {
            out.extend_from_slice(&data[child.start..child.end]);
        }
    }
    (out, fourcc)
}

// default_KID from tenc (moov → stsd → enca → sinf → schi → tenc)
const KID_PATH: &[&[u8; 4]] = &[b"trak", b"mdia", b"minf", b"stbl", b"sinf", b"schi"];

pub(crate) fn find_kid(data: &[u8]) -> Option<[u8; 16]> {
    let moov: Box = parse(data, 0, data.len()).into_iter().find(|b| b.is(b"moov"))?;
    find_kid_in(data, moov.content_start, moov.end, 0)
}

fn find_kid_in(data: &[u8], start: usize, end: usize, depth: usize) -> Option<[u8; 16]> {
    if depth > MAX_TREE_DEPTH { return None; }
    for child in parse(data, start, end) {
        match &child.typ {
            b"tenc"                       => return parse_tenc(data, child.content_start, child.end),
            typ if KID_PATH.contains(&typ) => { if let Some(k) = find_kid_in(data, child.content_start, child.end, depth + 1) { return Some(k); } }
            b"stsd"                       => { if let Some(k) = find_kid_in(data, child.content_start + 8, child.end, depth + 1) { return Some(k); } }
            b"enca" | b"mp4a"             => { if let Some(k) = find_kid_in(data, child.content_start + 28, child.end, depth + 1) { return Some(k); } }
            _                             => {}
        }
    }
    None
}

// tenc v0: KID at content+8; v1: at content+9 (pattern bytes shift it)
fn parse_tenc(data: &[u8], start: usize, end: usize) -> Option<[u8; 16]> {
    if end < start + 24 { return None; } // start+24, not end-start — callers may pass start > end
    let kid_off: usize = if data[start] == 0 { 8 } else { 9 };
    if start + kid_off + 16 > end { return None; }
    let mut kid: [u8; 16] = [0u8; 16];
    kid.copy_from_slice(&data[start + kid_off..start + kid_off + 16]);
    Some(kid)
}
