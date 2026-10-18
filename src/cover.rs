// Cover prep: oversized JPEGs → area-average downscale to 1000px, q90 re-encode;
// fitting covers and PNGs pass through zero-copy (no `image` crate, no external
// JPEG codec — decode and encode both live in crate::jpeg)

use std::borrow::Cow;

use crate::jpeg;
use crate::meta::Cover;
use crate::Err;

const MAX_EDGE: u32 = 1000;

pub(crate) fn is_jpeg(b: &[u8]) -> bool { b.len() >= 2 && b[0] == 0xFF && b[1] == 0xD8 }
pub(crate) fn is_png(b: &[u8]) -> bool { b.len() >= 8 && b[0..4] == [0x89, 0x50, 0x4E, 0x47] }

/// Validate and (when needed) downscale a cover image to fit MAX_EDGE.
pub(crate) fn prepare(raw: &[u8]) -> Result<Cover<'_>, Err> {
    if is_png(raw) {
        return Ok(Cover { data: Cow::Borrowed(raw), kind: 14 });
    }
    if !is_jpeg(raw) {
        return Err(Err::BadCover);
    }

    let (w, h, ch, pixels): (usize, usize, usize, Vec<u8>) = jpeg::decode(raw)?;
    if w <= MAX_EDGE as usize && h <= MAX_EDGE as usize {
        return Ok(Cover { data: Cow::Borrowed(raw), kind: 13 }); // fits — original bytes untouched
    }
    let (nw, nh): (u32, u32) = fit(w as u32, h as u32, MAX_EDGE);
    let scaled: Vec<u8> = resize_area(&pixels, w as u32, h as u32, ch, nw, nh);
    Ok(Cover { data: Cow::Owned(jpeg::encode(&scaled, nw as usize, nh as usize, ch)?), kind: 13 })
}

/// Longest-edge fit: (nw, nh) ≤ max_edge preserving aspect ratio.
fn fit(w: u32, h: u32, max_edge: u32) -> (u32, u32) {
    let long: u32 = w.max(h);
    let nw: u32 = ((u64::from(w) * u64::from(max_edge)) / u64::from(long)).max(1) as u32;
    let nh: u32 = ((u64::from(h) * u64::from(max_edge)) / u64::from(long)).max(1) as u32;
    (nw, nh)
}

// area-average (box) resample — clean anti-aliased downscale, hand-rolled
fn resize_area(src: &[u8], sw: u32, sh: u32, ch: usize, dw: u32, dh: u32) -> Vec<u8> {
    let (sw, sh, dw, dh): (usize, usize, usize, usize) = (sw as usize, sh as usize, dw as usize, dh as usize);
    let mut out: Vec<u8> = vec![0u8; dw * dh * ch];
    for dy in 0..dh {
        // source window [y0, y1) for this destination row
        let y0: usize = dy * sh / dh;
        let y1: usize = (((dy + 1) * sh) / dh).max(y0 + 1).min(sh);
        for dx in 0..dw {
            let x0: usize = dx * sw / dw;
            let x1: usize = (((dx + 1) * sw) / dw).max(x0 + 1).min(sw);
            for c in 0..ch {
                let mut sum: u64 = 0;
                for sy in y0..y1 {
                    let row: usize = sy * sw * ch;
                    for sx in x0..x1 {
                        sum += u64::from(src[row + sx * ch + c]);
                    }
                }
                let n: u64 = ((y1 - y0) * (x1 - x0)) as u64;
                out[(dy * dw + dx) * ch + c] = (sum / n) as u8;
            }
        }
    }
    out
}
