// iTunes ilst tags (title \xA9nam, date \xA9day, artists \xA9ART+aART, isrc ----,
// track trkn, disc disk, cover covr) — built inside the fused pass

use std::borrow::Cow;

use crate::boxes::{parse, write_header};
use crate::Err;

pub(crate) struct Cover {
    pub data: Vec<u8>,
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
            RawCover::Path(p)  => std::fs::read(p).map(Cow::Owned).map_err(|_| Err::InputNotFound)
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
    pub cover       : Option<RawCover>
}

const UTF8: u32 = 1;

fn data_atom(kind: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + payload.len());
    write_header(&mut out, b"data", 8 + payload.len());
    out.extend_from_slice(&kind.to_be_bytes()); // ilst type in the first byte
    out.extend_from_slice(&0u32.to_be_bytes()); // locale
    out.extend_from_slice(payload);
    out
}

fn item(fourcc: &[u8; 4], kind: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let d = data_atom(kind, payload);
    write_header(&mut out, fourcc, d.len());
    out.extend_from_slice(&d);
    out
}

fn text_item(fourcc: &[u8; 4], value: &str) -> Vec<u8> { item(fourcc, UTF8, value.as_bytes()) }

fn num_pair(n: u32, total: Option<u32>) -> Vec<u8> {
    let (v, t) = (n.min(0xFFFF) as u16, total.unwrap_or(0).min(0xFFFF) as u16);
    vec![0, 0, (v >> 8) as u8, v as u8, (t >> 8) as u8, t as u8, 0, 0]
}

// mean/name are standalone tag boxes (nested data atoms break every standards parser)
fn freeform_item(name: &str, value: &str) -> Vec<u8> {
    fn tag_box(typ: &[u8; 4], value: &str) -> Vec<u8> {
        let mut out = Vec::new();
        write_header(&mut out, typ, 4 + value.len());
        out.extend_from_slice(&UTF8.to_be_bytes());
        out.extend_from_slice(value.as_bytes());
        out
    }
    let mean = tag_box(b"mean", "com.apple.iTunes");
    let name_box = tag_box(b"name", name);
    let d = data_atom(UTF8, value.as_bytes());
    let inner = mean.len() + name_box.len() + d.len();
    let mut out = Vec::with_capacity(8 + inner);
    write_header(&mut out, b"----", inner);
    out.extend_from_slice(&mean);
    out.extend_from_slice(&name_box);
    out.extend_from_slice(&d);
    out
}

fn build_items(meta: &CoreMeta) -> Result<Vec<u8>, Err> {
    let mut out: Vec<u8> = Vec::new();
    if let Some(title) = &meta.title { out.extend_from_slice(&text_item(b"\xA9nam", title)); }
    if let Some(date) = &meta.date { out.extend_from_slice(&text_item(b"\xA9day", date)); }
    if let Some(artists) = &meta.artists {
        for artist in artists { out.extend_from_slice(&text_item(b"\xA9ART", artist)); }
        if let Some(first) = artists.first() { out.extend_from_slice(&text_item(b"aART", first)); }
    }
    if let Some(value) = &meta.isrc { out.extend_from_slice(&freeform_item("ISRC", value)); }
    if let Some(value) = &meta.album { out.extend_from_slice(&text_item(b"\xA9alb", value)); }
    if let Some(value) = &meta.genre { out.extend_from_slice(&text_item(b"\xA9gen", value)); }
    if let Some(value) = &meta.lyrics { out.extend_from_slice(&text_item(b"\xA9lyr", value)); }
    if let Some(value) = &meta.composer { out.extend_from_slice(&text_item(b"\xA9wrt", value)); }
    if let Some(num) = meta.track { out.extend_from_slice(&item(b"trkn", 0, &num_pair(num, meta.track_total))); }
    if let Some(num) = meta.disc { out.extend_from_slice(&item(b"disk", 0, &num_pair(num, meta.disc_total))); }
    if let Some(raw) = &meta.cover {
        let prepared = crate::cover::prepare(&raw.materialize()?)?;
        out.extend_from_slice(&item(b"covr", prepared.kind, &prepared.data));
    }
    Ok(out)
}

// metadata is written from scratch: any pre-existing ilst is wiped and
// rebuilt solely from the caller's TuneMeta (deterministic, no merging)
fn fresh_ilst(meta: &CoreMeta) -> Result<Vec<u8>, Err> {
    build_items(meta)
}

fn meta_box(ilst: &[u8]) -> Vec<u8> {
    // hdlr in full ISO form (with pre_defined) — the Windows handler bails on short ones
    let mut hdlr: Vec<u8> = Vec::new();
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&[0u8; 4]); // pre_defined
    payload.extend_from_slice(b"mdir");
    payload.extend_from_slice(b"appl");
    payload.extend_from_slice(&[0u8; 8]); // reserved
    payload.push(0);                      // empty name
    write_header(&mut hdlr, b"hdlr", payload.len() + 4);
    hdlr.extend_from_slice(&0u32.to_be_bytes());
    hdlr.extend_from_slice(&payload);

    let mut out: Vec<u8> = Vec::with_capacity(12 + hdlr.len() + ilst.len());
    write_header(&mut out, b"meta", 4 + hdlr.len() + ilst.len() + 8);
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&hdlr);
    write_header(&mut out, b"ilst", ilst.len());
    out.extend_from_slice(ilst);
    out
}

// rebuild moov so a freshly built udta/meta/ilst is the first child; any
// pre-existing udta (with its ilst) is removed entirely
pub(crate) fn rebuild_moov_with_meta(children: &[u8], meta: &CoreMeta) -> Result<Vec<u8>, Err> {
    // every old udta goes — keeping any would both leak stale tags and shadow the
    // fresh subtree for the Windows handler (it reads only the first udta/meta)
    let udta: Vec<(usize, usize)> = parse(children, 0, children.len())
        .iter().filter(|c| c.is(b"udta")).map(|c| (c.start, c.end)).collect();

    let new_meta: Vec<u8> = meta_box(&fresh_ilst(meta)?);

    // udta goes first in moov — the Windows MP4 handler reads only the first udta/meta subtree
    let mut out: Vec<u8> = Vec::with_capacity(children.len() + new_meta.len());
    write_header(&mut out, b"udta", new_meta.len());
    out.extend_from_slice(&new_meta);
    let mut pos: usize = 0;
    for (start, end) in udta {
        out.extend_from_slice(&children[pos..start]);
        pos = end;
    }
    out.extend_from_slice(&children[pos..]);
    Ok(out)
}
