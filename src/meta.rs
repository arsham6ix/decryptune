// iTunes ilst tags (title \xA9nam, date \xA9day, artists \xA9ART+aART, isrc ----,
// track trkn, disc disk, cover covr) — built inside the fused pass

use std::borrow::Cow;

use crate::boxes::{parse, write_header, Box, Writer};
use crate::Err;

pub(crate) struct Cover<'a> {
    pub data: Cow<'a, [u8]>,
    pub kind: u32 // ilst data-type: 13 = JPEG, 14 = PNG
}

// cover as received from Python — the disk read happens in the worker, never
// under the GIL (a slow cover path must not stall the caller's event loop)
pub(crate) enum RawCover {
    Bytes(Vec<u8>),
    Path(String)
}

impl RawCover {
    fn materialize(&self) -> Result<Cow<'_, [u8]>, Err> {
        match self {
            RawCover::Bytes(b) => Ok(Cow::Borrowed(b)),
            RawCover::Path(p)  => std::fs::read(p).map(Cow::Owned).map_err(|_| Err::CoverNotFound)
        }
    }
}

pub(crate) struct CoreMeta {
    pub title       : Option<String>,
    pub date        : Option<String>,
    pub artists     : Option<Vec<String>>,
    pub isrc        : Option<String>,
    pub track       : Option<u32>,
    pub track_total : Option<u32>,
    pub disc        : Option<u32>,
    pub disc_total  : Option<u32>,
    pub album       : Option<String>,
    pub genre       : Option<String>,
    pub lyrics      : Option<String>,
    pub composer    : Option<String>,
    pub cover       : Option<RawCover>,
    pub cover_ready : Option<(Vec<u8>, u32)> // prepared once, shared by measure + build
}

// resolve the cover exactly once (disk read + JPEG decode/resize/encode) so
// the measure and build passes share a single codec run
pub(crate) fn pre_resolve_cover(meta: &mut CoreMeta) -> Result<(), Err> {
    if meta.cover_ready.is_none() {
        if let Some(raw) = &meta.cover {
            let bytes: Cow<'_, [u8]> = raw.materialize()?;
            let prepared: Cover<'_> = crate::cover::prepare(&bytes)?;
            meta.cover_ready = Some((prepared.data.into_owned(), prepared.kind));
        }
    }
    Ok(())
}

const UTF8: u32 = 1;

// iTunes tag items are written straight into the ilst buffer — no per-tag
// intermediate allocation
fn push_item(out: &mut Vec<u8>, fourcc: &[u8; 4], kind: u32, payload: &[u8]) {
    write_header(out, fourcc, 16 + payload.len());
    write_header(out, b"data", 8 + payload.len());
    out.extend_from_slice(&kind.to_be_bytes()); // ilst type in the first byte
    out.extend_from_slice(&0u32.to_be_bytes()); // locale
    out.extend_from_slice(payload);
}

fn push_text(out: &mut Vec<u8>, fourcc: &[u8; 4], value: &str) {
    push_item(out, fourcc, UTF8, value.as_bytes());
}

fn num_pair(n: u32, total: Option<u32>) -> [u8; 8] {
    let (v, t): (u16, u16) = (n.min(0xFFFF) as u16, total.unwrap_or(0).min(0xFFFF) as u16);
    [0, 0, (v >> 8) as u8, v as u8, (t >> 8) as u8, t as u8, 0, 0]
}

// mean/name are standalone tag boxes (nested data atoms break every standards parser)
fn push_freeform(out: &mut Vec<u8>, name: &str, value: &str) {
    fn tag_box(out: &mut Vec<u8>, typ: &[u8; 4], value: &str) {
        write_header(out, typ, 4 + value.len());
        out.extend_from_slice(&UTF8.to_be_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    let mut d: Vec<u8> = Vec::with_capacity(16 + value.len());
    write_header(&mut d, b"data", 8 + value.len());
    d.extend_from_slice(&UTF8.to_be_bytes());
    d.extend_from_slice(&0u32.to_be_bytes());
    d.extend_from_slice(value.as_bytes());
    // mean (8+4+16) + name (8+4+len) + data item
    let inner: usize = 28 + 12 + name.len() + d.len();
    write_header(out, b"----", inner);
    tag_box(out, b"mean", "com.apple.iTunes");
    tag_box(out, b"name", name);
    out.extend_from_slice(&d);
}

fn build_items(out: &mut Vec<u8>, meta: &CoreMeta) -> Result<(), Err> {
    if let Some(title) = &meta.title { push_text(out, b"\xA9nam", title); }
    if let Some(date) = &meta.date { push_text(out, b"\xA9day", date); }
    if let Some(artists) = &meta.artists {
        for artist in artists { push_text(out, b"\xA9ART", artist); }
        if let Some(first) = artists.first() { push_text(out, b"aART", first); }
    }
    if let Some(value) = &meta.isrc { push_freeform(out, "ISRC", value); }
    if let Some(value) = &meta.album { push_text(out, b"\xA9alb", value); }
    if let Some(value) = &meta.genre { push_text(out, b"\xA9gen", value); }
    if let Some(value) = &meta.lyrics { push_text(out, b"\xA9lyr", value); }
    if let Some(value) = &meta.composer { push_text(out, b"\xA9wrt", value); }
    if let Some(num) = meta.track { push_item(out, b"trkn", 0, &num_pair(num, meta.track_total)); }
    if let Some(num) = meta.disc { push_item(out, b"disk", 0, &num_pair(num, meta.disc_total)); }
    if let Some((data, kind)) = &meta.cover_ready {
        push_item(out, b"covr", *kind, data);
    } else if let Some(raw) = &meta.cover {
        let bytes: Cow<'_, [u8]> = raw.materialize()?;
        let prepared: Cover<'_> = crate::cover::prepare(&bytes)?;
        push_item(out, b"covr", prepared.kind, prepared.data.as_ref());
    }
    Ok(())
}

// hdlr in full ISO form (with pre_defined) — the Windows handler bails on short ones
fn hdlr_box() -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&[0u8; 4]); // pre_defined
    payload.extend_from_slice(b"mdir");
    payload.extend_from_slice(b"appl");
    payload.extend_from_slice(&[0u8; 8]); // reserved
    payload.push(0);                      // empty name
    let mut hdlr: Vec<u8> = Vec::with_capacity(12 + payload.len());
    write_header(&mut hdlr, b"hdlr", payload.len() + 4);
    hdlr.extend_from_slice(&0u32.to_be_bytes());
    hdlr.extend_from_slice(&payload);
    hdlr
}

// metadata is written from scratch: any pre-existing ilst is wiped and rebuilt
// solely from the caller's TuneMeta (deterministic, no merging); the fresh
// udta/meta/ilst subtree goes first in moov — any pre-existing udta (with its
// ilst) is removed entirely. The whole moov is appended to the writer in one
// shot (no intermediate content buffer).
pub(crate) fn append_moov_with_meta(out: &mut Writer<'_>, children: &[u8], meta: &CoreMeta) -> Result<(), Err> {
    // every old udta goes — keeping any would both leak stale tags and shadow the
    // fresh subtree for the Windows handler (it reads only the first udta/meta)
    let top: Vec<Box> = parse(children, 0, children.len());
    if top.len() >= crate::boxes::MAX_BOXES {
        return Err(Err::InvalidMp4); // saturated parse: the udta map would be wrong
    }
    let udta: Vec<(usize, usize)> = top.iter().filter(|c| c.is(b"udta"))
        .map(|c| (c.start, c.end)).collect();

    let mut ilst: Vec<u8> = Vec::new();
    build_items(&mut ilst, meta)?;
    let hdlr: Vec<u8> = hdlr_box();
    let meta_content: usize = 4 + hdlr.len() + 8 + ilst.len(); // version/flags + hdlr + ilst
    let removed: usize = udta.iter().map(|(s, e)| e - s).sum();

    out.header(b"moov", 8 + 8 + meta_content + children.len() - removed);
    // udta goes first in moov — the Windows MP4 handler reads only the first udta/meta subtree
    out.header(b"udta", 8 + meta_content);
    out.header(b"meta", meta_content);
    out.put(&0u32.to_be_bytes());
    out.put(&hdlr);
    out.header(b"ilst", ilst.len());
    out.put(&ilst);

    let mut pos: usize = 0;
    for (start, end) in &udta {
        out.put(&children[pos..*start]);
        pos = *end;
    }
    out.put(&children[pos..]);
    Ok(())
}
