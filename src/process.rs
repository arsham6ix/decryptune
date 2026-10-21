use std::ops::Range;

use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};

use crate::boxes::{header_len, parse, write_header, Box, Writer, MAX_BOXES};
use crate::meta::{append_moov_with_meta, CoreMeta};
use crate::Err;

type Aes128Ctr = ctr::Ctr128BE<Aes128>;

const CONTAINERS: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl"];
const MAX_TREE_DEPTH: usize = 16;
// chi2 thresholds calibrated on real AAC tracks (structured ~3800..14000 at
// 16KB, ~950+ at 4KB; noise is df-bound at ~210..340 for any size above ~1KB,
// so 500 splits clean from corrupt with margin on both sides down to 4KB)
const MIN_JUDGE_BYTES: usize = 4 * 1024;
const CHI2_CORRUPT_BELOW: f64 = 500.0;
// hostile default-fill truns claim up to 2^32 samples for a few bytes of box;
// each claimed sample costs a u32 of sizes storage, so a handful of such truns
// in one traf would multiply the file size into an allocation abort. No real
// fragment carries anywhere near this many samples.
const MAX_SAMPLES: usize = 1 << 22; // 4M samples per fragment
// fragment work is spread over threads only when it is large enough to pay
// for the spawns (a small single-fragment track stays on one thread)
const MIN_PAR_BYTES: usize = 8 * 1024 * 1024;

#[allow(dead_code)] // fields beyond `corrupt` are diagnostics for future reporting
pub(crate) struct FragmentStat {
    pub index     : usize,
    pub encrypted : bool,
    pub bytes     : usize,
    pub chi2      : f64,
    pub corrupt   : Option<bool> // Some(true)=corrupt, Some(false)=clean, None=not judged
}

// one fragment's phase-2 work: decrypt + judge its own payload range, disjoint
// from every other job, so the ranges parallelize safely
struct Job {
    range    : Range<usize>, // payload span inside the final output buffer
    ivs      : Option<Vec<[u8; 16]>>,
    sizes    : Vec<u32>,
    default  : Option<u32>,
    encrypted: bool,
    judged   : bool
}

struct Plan {
    ivs         : Option<Vec<[u8; 16]>>,
    sizes       : Vec<u32>,
    default     : Option<u32>,
    ok          : bool,
    rebuilt_len : usize
}

/// Exact output length — the same fused walk as `build`, counting bytes
/// instead of writing (no payload copy, no decrypt). Cheap: O(parse).
pub fn measure(data: &[u8], meta: Option<&CoreMeta>) -> Result<usize, Err> {
    let mut w: Writer<'_> = Writer::Count(0);
    walk(data, None, false, meta, &mut w)?;
    Ok(w.len())
}

/// Run the fused pass into an exactly-sized buffer (`measure` provides the
/// size). Returns one stat per decrypted fragment, in fragment order.
pub fn build(data: &[u8], key: Option<&[u8; 16]>, strict: bool, meta: Option<&CoreMeta>, buf: &mut [u8]) -> Result<Vec<FragmentStat>, Err> {
    let mut w: Writer<'_> = Writer::Buf { buf, pos: 0 };
    let jobs: Vec<Job> = walk(data, key, strict, meta, &mut w)?;
    let Writer::Buf { buf, .. } = w else { unreachable!("build always writes a Buf") };
    Ok(run_jobs(buf, jobs, key))
}

// fused single-walk: decrypt-plan + sanitize + metadata in one pass over the
// boxes. Measure mode runs the identical control flow — Writer::Count turns
// every payload copy into a size add, so the length is exact by construction
fn walk(data: &[u8], key: Option<&[u8; 16]>, strict: bool, meta: Option<&CoreMeta>, out: &mut Writer<'_>) -> Result<Vec<Job>, Err> {
    let boxes: Vec<Box> = parse(data, 0, data.len());
    if boxes.is_empty() || boxes.len() >= MAX_BOXES { return Err(Err::InvalidMp4); }

    let mut deltas: Vec<i64> = Vec::new();
    let mut patches: Vec<(usize, Vec<usize>)> = Vec::new(); // (frag base, sidx word offsets)
    let mut jobs: Vec<Job> = Vec::new();
    let mut pending: Option<Range<usize>> = None; // queued passthrough run
    let mut unpaired_moof: bool = false;

    let mut i = 0;
    while i < boxes.len() {
        let b: &Box = &boxes[i];
        match &b.typ {
            b"moof" => {
                flush(out, data, &mut pending);
                let plan: Plan = visit_moof(out, data, b);
                deltas.push((b.end - b.start) as i64 - plan.rebuilt_len as i64);
                if i + 1 < boxes.len() && boxes[i + 1].is(b"mdat") {
                    if let Some(job) = stage_mdat(out, data, &boxes[i + 1], plan, key, strict, unpaired_moof) {
                        jobs.push(job);
                    }
                    unpaired_moof = false;
                    i += 2;
                } else {
                    // this fragment's payload is not adjacent — if a later mdat
                    // follows another moof, that mdat may hold samples of several
                    // fragments and no plan can be trusted on it
                    unpaired_moof = true;
                    i += 1;
                }
                continue;
            }
            b"moov" => {
                flush(out, data, &mut pending);
                let children: Vec<u8> = rebuild_container(data, b.content_start, b.end, 0);
                if let Some(m) = meta {
                    // chunk offsets point into mdat — growing moov breaks them only
                    // when an mdat sits after moov or an earlier moof rebuild shifted
                    // the data; in the moov-at-end layout the write is safe even
                    // with a non-empty stco
                    let mdat_after: bool = boxes[i + 1..].iter().any(|x| x.is(b"mdat"));
                    let shifted: bool = deltas.iter().any(|d| *d != 0);
                    if has_chunk_offsets(&children, 0) && (mdat_after || shifted) {
                        return Err(Err::MetaNeedsFragmented);
                    }
                    append_moov_with_meta(out, &children, m)?;
                } else {
                    out.header(b"moov", children.len());
                    out.put(&children);
                }
            }
            b"sidx" => {
                flush(out, data, &mut pending);
                let base: usize = out.len();
                out.put(&data[b.start..b.end]);
                patches.push((deltas.len(), sidx_words(data, b, base)));
            }
            _ => {
                // queue passthrough boxes into contiguous runs — one memcpy per run
                pending = match pending.take() {
                    Some(r) if r.end == b.start => Some(r.start..b.end),
                    Some(r) => { out.put(&data[r.start..r.end]); Some(b.start..b.end) }
                    None    => Some(b.start..b.end)
                };
            }
        }
        i += 1;
    }
    flush(out, data, &mut pending);

    // patch sidx referenced sizes in place, now that fragment deltas are known
    if let Writer::Buf { buf, pos } = &mut *out {
        patch_sidx(&mut buf[..*pos], &deltas, &patches);
    }
    Ok(jobs)
}

// emit a queued passthrough run as one memcpy instead of per-box copies
fn flush(out: &mut Writer<'_>, data: &[u8], pending: &mut Option<Range<usize>>) {
    if let Some(r) = pending.take() { out.put(&data[r.start..r.end]); }
}

fn visit_moof(out: &mut Writer<'_>, data: &[u8], b: &Box) -> Plan {
    let mut content: Vec<u8> = Vec::new();
    let mut ivs: Option<Vec<[u8; 16]>> = None;
    let mut sizes: Vec<u32> = Vec::new();
    let mut default_size: Option<u32> = None;
    let mut oversized: bool = false;
    let mut subsampled: bool = false;

    for child in parse(data, b.content_start, b.end) {
        if !child.is(b"traf") {
            content.extend_from_slice(&data[child.start..child.end]);
            continue;
        }
        let traf: Vec<Box> = parse(data, child.content_start, child.end);

        // gather: tfhd default first so trun fallback works in any order
        // (a saturated child count means box spam — nothing to decrypt)
        if traf.len() >= MAX_BOXES { oversized = true; }
        default_size = traf.iter().find(|t| t.is(b"tfhd"))
            .and_then(|t| parse_tfhd(data, t.content_start, t.end));
        for t in &traf {
            match &t.typ {
                b"senc" if ivs.is_none() => {
                    if let Some((list, subs)) = parse_senc(data, t.content_start, t.end) {
                        ivs = Some(list);
                        subsampled = subs;
                    }
                }
                b"trun" if !oversized    => {
                    sizes.extend(parse_trun_sizes(data, t.content_start, t.end, default_size));
                    if sizes.len() > MAX_SAMPLES { // many capped truns must not sum back into a bomb
                        sizes.clear();
                        oversized = true;
                    }
                }
                b"trun"                  => {}
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
                    write_traf_trun(&mut traf_out, data, t, removed);
                }
                _ => traf_out.extend_from_slice(&data[t.start..t.end])
            }
        }
        write_header(&mut content, b"traf", traf_out.len());
        content.extend_from_slice(&traf_out);
    }

    // senc/trun count mismatch → partially plain fragment (part of the track is
    // stored in the clear); skip decryption for it. An oversized trun claim is
    // treated the same way, and so is real subsample encryption — per-entry
    // clear/protected ranges are not applied, so full-sample decryption would
    // corrupt the clear bytes; the fragment passes through untouched instead.
    let ok: bool = !oversized && !subsampled
        && matches!(&ivs, Some(list) if !sizes.is_empty() && list.len() == sizes.len());
    out.header(b"moof", content.len());
    out.put(&content);
    // the rebuilt moof's header may itself be largesize — the delta must count
    // exactly what the writer emitted
    Plan { ivs, sizes, default: default_size, ok, rebuilt_len: content.len() + header_len(content.len()) }
}

fn write_traf_trun(traf_out: &mut Vec<u8>, data: &[u8], t: &Box, removed: i64) {
    write_header(traf_out, b"trun", t.end - t.content_start);
    traf_out.extend_from_slice(&patch_trun(data, t.content_start, t.end, removed));
}

// chi2 costs a full pass over the payload — paid per fragment in phase 2.
// The counting phase is factored out so parallel chunks can merge histograms
// additively and the statistic is computed once per fragment.
fn hist(buf: &[u8]) -> [u64; 256] {
    let mut lanes: [[u64; 256]; CHI2_LANES] = [[0u64; 256]; CHI2_LANES];
    let (body, tail): (&[u8], &[u8]) = buf.split_at(buf.len() & !(CHI2_LANES - 1));
    for c in body.chunks_exact(CHI2_LANES) {
        for (l, &b) in c.iter().enumerate() { lanes[l][b as usize] += 1; }
    }
    for &b in tail { lanes[0][b as usize] += 1; }
    let mut total: [u64; 256] = [0u64; 256];
    for lane in &lanes {
        for (t, &o) in total.iter_mut().zip(lane.iter()) { *t += o; }
    }
    total
}

fn chi2_hist(h: &[u64; 256], n: usize) -> f64 {
    if n == 0 { return 0.0; }
    let e: f64 = n as f64 / 256.0;
    h.iter().map(|&o| { let d: f64 = o as f64 - e; d * d / e }).sum()
}

fn chi2(buf: &[u8]) -> f64 {
    let n: usize = buf.len();
    if n == 0 { return 0.0; }
    chi2_hist(&hist(buf), n)
}

// 16 sub-histograms break the same-bucket increment dependency chain (16
// independent chains instead of one) — the scalar loop was the whole strict cost
const CHI2_LANES: usize = 16;

fn stage_mdat(out: &mut Writer<'_>, data: &[u8], m: &Box, plan: Plan, key: Option<&[u8; 16]>, strict: bool, ambiguous: bool) -> Option<Job> {
    let content_len: usize = m.end - m.content_start;
    // a largesize (64-bit) mdat keeps its 64-bit header — a u32 rewrite would wrap
    if content_len + 8 > u32::MAX as usize || u32::from_be_bytes(data[m.start..m.start + 4].try_into().unwrap()) == 1 {
        out.put(&1u32.to_be_bytes());
        out.put(b"mdat");
        out.put(&((content_len as u64) + 16).to_be_bytes());
    } else {
        out.header(b"mdat", content_len);
    }
    let base: usize = out.len();
    out.put(&data[m.content_start..m.end]);
    // measure mode records no jobs — phase 2 never runs there
    let Writer::Buf { .. } = out else { return None };
    Some(Job {
        range    : base..out.len(),
        ivs      : plan.ivs,
        sizes    : plan.sizes,
        default  : plan.default,
        encrypted: plan.ok && !ambiguous,
        judged   : plan.ok && !ambiguous && key.is_some() && strict && content_len >= MIN_JUDGE_BYTES
    })
}

// phase 2: decrypt + judge every fragment. Small workloads stay on the calling
// thread; large ones split across worker threads over disjoint payload spans —
// across fragments when there are many, within a fragment when one is huge
// (histograms merge additively, so chi2 is identical to the serial pass)
fn run_jobs(buf: &mut [u8], jobs: Vec<Job>, key: Option<&[u8; 16]>) -> Vec<FragmentStat> {
    if jobs.is_empty() { return Vec::new(); }
    let total: usize = jobs.iter().map(|j| j.range.len()).sum();
    let cores: usize = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if !key.is_some() || total < MIN_PAR_BYTES || cores == 1 {
        return jobs.iter().map(|job| decrypt_job(&mut buf[job.range.clone()], job, key)).collect();
    }

    let chunks: Vec<Chunk<'_>> = jobs.iter().enumerate()
        .flat_map(|(i, job)| job_chunks(i, job, (cores * job.range.len()).div_ceil(total)))
        .collect();
    // the buffer base crosses threads as a usize — disjoint capture would
    // otherwise hoist the raw pointer out of any wrapper and break Send.
    // Each worker folds its same-job chunk runs into one Verdict, so nothing
    // per-chunk crosses back except split-job partials
    let base: usize = buf.as_ptr() as usize;
    let folded: Vec<Vec<(usize, Verdict)>> = std::thread::scope(|s| {
        let per: usize = chunks.len().div_ceil(cores).max(1);
        let handles: Vec<_> = chunks.chunks(per)
            .map(|group| s.spawn(move || -> Vec<(usize, Verdict)> {
                let mut out: Vec<(usize, Verdict)> = Vec::new();
                let mut i: usize = 0;
                while i < group.len() {
                    let j: usize = group[i].job;
                    let first_start: usize = group[i].start;
                    let mut acc: [u64; 256] = [0u64; 256];
                    let mut complete: bool = false;
                    while i < group.len() && group[i].job == j {
                        let c: &Chunk<'_> = &group[i];
                        if let Some(h) = run_chunk(unsafe {
                            std::slice::from_raw_parts_mut((base + c.start) as *mut u8, c.len)
                        }, c, key) {
                            for (a, &o) in acc.iter_mut().zip(h.iter()) { *a += o; }
                        }
                        complete = c.last;
                        i += 1;
                    }
                    let bytes: usize = group[i - 1].start + group[i - 1].len - first_start;
                    out.push((j, if complete {
                        Verdict::Chi2(chi2_hist(&acc, bytes))
                    } else {
                        Verdict::Partial(acc.into())
                    }));
                }
                out
            }))
            .collect();
        handles.into_iter().map(|h| h.join().expect("decrypt worker panicked")).collect()
    });

    let mut partials: Vec<Option<[u64; 256]>> = Vec::new();
    let mut stats: Vec<FragmentStat> = jobs.iter().map(|job| FragmentStat {
        index: 0, encrypted: job.encrypted, bytes: job.range.len(), chi2: f64::NAN,
        corrupt: if !job.encrypted { Some(false) } else if !job.judged { None } else { Some(false) }
    }).collect();
    for group in &folded {
        for (j, v) in group {
            match v {
                Verdict::Chi2(c) => stats[*j].chi2 = *c,
                Verdict::Partial(h) => {
                    if jobs[*j].judged {
                        while partials.len() <= *j { partials.push(None); }
                        match &mut partials[*j] {
                            Some(acc) => for (a, &o) in acc.iter_mut().zip(h.iter()) { *a += o; },
                            slot @ None => *slot = Some(*h.clone()),
                        }
                    }
                }
            }
        }
    }
    for (j, job) in jobs.iter().enumerate() {
        if job.judged {
            // NaN = no complete verdict landed in any worker (or the job's
            // chunks split across workers — its partial was merged instead)
            let chi2: f64 = if stats[j].chi2.is_nan() {
                partials.get(j).and_then(|h| h.as_ref()).map_or(0.0, |h| chi2_hist(h, job.range.len()))
            } else {
                stats[j].chi2
            };
            stats[j].chi2 = chi2;
            stats[j].corrupt = Some(chi2 < CHI2_CORRUPT_BELOW);
        }
    }
    for (i, f) in stats.iter_mut().enumerate() { f.index = i; }
    stats
}

// one byte span of a fragment: disjoint from every other chunk's span, so the
// AES writes and the chi2 histogram never cross threads. `sample` is the resume
// point — the sample containing the span's first byte and the offset inside it
// (CTR keystream blocks are addressable, so a span may start mid-sample) — or
// None when the span holds only ciphertext tail. `live_len` caps decryption at
// the serial loop's break point; the rest of the span is untouched payload.
// `last` marks the fragment's final chunk so a worker can fold per-job results.
struct Chunk<'a> {
    job    : usize,
    start  : usize, // absolute byte offset inside the output buffer
    len    : usize,
    judged : bool,
    last   : bool,
    sample : Option<(usize, usize)>,
    live_len: usize,
    ivs    : &'a [[u8; 16]],
    sizes  : &'a [u32],
    default: Option<u32>
}

// a worker's folded result for one job: the final statistic when the job's
// chunks all landed in this group, or a partial histogram to merge upstream
// (only jobs split across group boundaries carry one — rare, hence boxed)
enum Verdict {
    Chi2(f64),
    Partial(std::boxed::Box<[u64; 256]>),
}

// split one fragment into `want` byte spans. The serial loop's break rule (the
// first zero-size or overflowing sample ends the fragment's stream) is baked
// into the live prefix; spans past it carry no resume point and only histogram
fn job_chunks<'a>(idx: usize, job: &'a Job, want: usize) -> Vec<Chunk<'a>> {
    let plain = |start: usize, len: usize, sample: Option<(usize, usize)>, live_len: usize| Chunk {
        job: idx, start, len, judged: job.judged, last: false, sample, live_len,
        ivs: job.ivs.as_deref().unwrap_or(&[]), sizes: &job.sizes, default: job.default
    };
    if !job.encrypted {
        return vec![plain(job.range.start, job.range.len(), None, 0)];
    }
    let want = want.max(1);
    if want == 1 { // common case: the whole fragment on one worker, serial rules
        let live = live_prefix(job);
        return vec![plain(job.range.start, job.range.len(), Some((0, 0)), live)];
    }
    let live_end: usize = live_prefix(job);
    let ivs: &[[u8; 16]] = job.ivs.as_deref().unwrap_or(&[]);
    // byte offset of each live sample's start (live_prefix guarantees the same
    // break rule, so offsets are strictly increasing below live_end)
    let mut offsets: Vec<usize> = Vec::new();
    let mut pos: usize = 0;
    for &sz in job.sizes.iter().take(ivs.len()) {
        let sz: usize = if sz == 0 { job.default.unwrap_or(0) } else { sz } as usize;
        if sz == 0 || pos + sz > job.range.len() { break; }
        offsets.push(pos);
        pos += sz;
    }
    let span: usize = job.range.len().div_ceil(want);
    let mut chunks: Vec<Chunk<'a>> = Vec::new();
    for k in 0..want {
        let s: usize = k * span;
        if s >= job.range.len() { break; }
        let len: usize = span.min(job.range.len() - s);
        let sample: Option<(usize, usize)> = if s >= live_end {
            None
        } else {
            let si: usize = offsets.partition_point(|&o| o <= s).max(1) - 1;
            Some((si, s - offsets[si]))
        };
        chunks.push(plain(job.range.start + s, len, sample, live_end.saturating_sub(s).min(len)));
    }
    if let Some(c) = chunks.last_mut() { c.last = true; }
    chunks
}

// byte length of the serial decrypt loop's live prefix for one fragment
fn live_prefix(job: &Job) -> usize {
    let ivs: &[[u8; 16]] = job.ivs.as_deref().unwrap_or(&[]);
    let mut pos: usize = 0;
    for (iv, &sz) in ivs.iter().zip(job.sizes.iter()) {
        let _ = iv;
        let sz: usize = if sz == 0 { job.default.unwrap_or(0) } else { sz } as usize;
        if sz == 0 || pos + sz > job.range.len() { break; }
        pos += sz;
    }
    pos
}

// decrypt one chunk's span (resuming mid-sample via the CTR counter) and
// histogram it when judged
fn run_chunk(buf: &mut [u8], chunk: &Chunk<'_>, key: Option<&[u8; 16]>) -> Option<[u64; 256]> {
    if let (Some(k), Some((mut s, mut skip))) = (key, chunk.sample) {
        let mut pos: usize = 0;
        while pos < chunk.live_len && s < chunk.sizes.len() {
            let sz: usize = if chunk.sizes[s] == 0 { chunk.default.unwrap_or(0) } else { chunk.sizes[s] } as usize;
            if sz == 0 { break; } // serial break rule — a live prefix never contains one
            let take: usize = (sz - skip).min(chunk.live_len - pos).min(buf.len() - pos);
            let mut cipher: Aes128Ctr = Aes128Ctr::new(k.into(), (&chunk.ivs[s]).into());
            if skip > 0 { cipher.seek(skip as u64); }
            cipher.apply_keystream(&mut buf[pos..pos + take]);
            pos += take;
            skip = 0;
            s += 1;
        }
    }
    if chunk.judged { Some(hist(buf)) } else { None }
}

fn decrypt_job(buf: &mut [u8], job: &Job, key: Option<&[u8; 16]>) -> FragmentStat {
    if job.encrypted {
        if let (Some(ivs), Some(key)) = (&job.ivs, key) {
            let mut pos: usize = 0;
            for (iv, &sz) in ivs.iter().zip(job.sizes.iter()) {
                // a zero entry in a per-entry table means "use the tfhd default"
                let sz: usize = if sz == 0 { job.default.unwrap_or(0) } else { sz } as usize;
                if sz == 0 || pos + sz > buf.len() { break; } // continuing would misalign every later sample
                Aes128Ctr::new(key.into(), iv.into()).apply_keystream(&mut buf[pos..pos + sz]);
                pos += sz;
            }
        }
    }
    let chi2: f64 = if job.judged { chi2(buf) } else { 0.0 };
    let corrupt = if !job.encrypted { Some(false) } else if !job.judged { None } else { Some(chi2 < CHI2_CORRUPT_BELOW) };
    FragmentStat { index: 0, encrypted: job.encrypted, bytes: buf.len(), chi2, corrupt }
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
            typ if CONTAINERS.contains(&typ)
                && has_chunk_offsets(&children[child.content_start..child.end], depth + 1) => return true,
            _ => {}
        }
    }
    false
}

// default_sample_size from tfhd (flag 0x10), honoring preceding optional fields;
// every read stays inside the tfhd box — never spills into a sibling box
fn parse_tfhd(data: &[u8], start: usize, end: usize) -> Option<u32> {
    if start + 8 > end { return None; }
    let flags: u32 = u32::from_be_bytes(data[start..start + 4].try_into().ok()?) & 0xFFFFFF;
    let mut pos: usize = start + 8; // version/flags + track_ID
    if flags & 0x1 != 0 { pos += 8; }  // base_data_offset
    if flags & 0x2 != 0 { pos += 4; }  // sample_description_index
    if flags & 0x8 != 0 { pos += 4; }  // default_sample_duration
    if flags & 0x10 == 0 || pos + 4 > end { return None; }
    Some(u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?))
}

// IVs from senc plus whether real subsample entries exist (flags 0x2 with a
// non-empty per-entry table): subsample ranges are parsed for IV alignment
// only — bounded by the box end, never the file
fn parse_senc(data: &[u8], start: usize, end: usize) -> Option<(Vec<[u8; 16]>, bool)> {
    if start + 8 > end { return None; }
    let flags: u32 = u32::from_be_bytes(data[start..start + 4].try_into().ok()?) & 0xFFFFFF;
    let count: usize = u32::from_be_bytes(data[start + 4..start + 8].try_into().ok()?) as usize;
    if count > (end - start - 8) / 8 { return None; } // hostile count bound
    let mut pos: usize = start + 8;
    let mut ivs: Vec<[u8; 16]> = Vec::with_capacity(count);
    let mut subsampled: bool = false;
    for _ in 0..count {
        if pos + 8 > end { return None; }
        let mut iv: [u8; 16] = [0u8; 16];
        iv[..8].copy_from_slice(&data[pos..pos + 8]);
        pos += 8;
        if flags & 0x2 != 0 {
            if pos + 2 > end { return None; }
            let sub: usize = u16::from_be_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            if sub > 0 { subsampled = true; }
            pos += 2 + sub * 6;
        }
        ivs.push(iv);
    }
    Some((ivs, subsampled))
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
    // fill cannot exceed a quarter of the file (one u32 size each) nor the
    // per-fragment sample cap
    let count: usize = if flags & 0x200 != 0 && stride != 0 {
        count.min(end.saturating_sub(pos) / stride) // pos may sit past end on hostile flags
    } else {
        count.min(data.len() / 4).min(MAX_SAMPLES)
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

// referenced_size words shrink by each fragment's rebuild delta; out-of-range
// or non-shrinking entries are left untouched
fn patch_sidx(out: &mut [u8], deltas: &[i64], patches: &[(usize, Vec<usize>)]) {
    for (frag_base, words) in patches {
        for (k, &off) in words.iter().enumerate() {
            let fi: usize = frag_base + k;
            if fi >= deltas.len() || deltas[fi] <= 0 || off + 4 > out.len() { continue; }
            let d: u32 = (deltas[fi] as u64).min(0x7FFF_FFFF) as u32;
            let word: u32 = u32::from_be_bytes(out[off..off + 4].try_into().unwrap());
            let new: u32 = (word & 0x7FFF_FFFF).saturating_sub(d);
            out[off..off + 4].copy_from_slice(&((word & 0x8000_0000) | (new & 0x7FFF_FFFF)).to_be_bytes());
        }
    }
}

fn rebuild_container(data: &[u8], start: usize, end: usize, depth: usize) -> Vec<u8> {
    let children: Vec<Box> = parse(data, start, end);
    if children.len() >= MAX_BOXES {
        // box spam: skip sanitizing this container entirely — the raw bytes
        // pass through and any stray pssh inside stays (a 2M-child moov is
        // hostile by definition; sanitizing it is not worth the memory)
        return data[start..end].to_vec();
    }
    let mut out: Vec<u8> = Vec::new();
    for child in &children {
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
