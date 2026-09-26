// Cover prep: oversized JPEGs → area-average downscale to 1000px, q90 re-encode;
// fitting covers and PNGs pass through byte-identical (minimal deps, no `image` crate)

use jpeg_decoder::{Decoder, ImageInfo, PixelFormat};
use jpeg_encoder::{Encoder, ColorType};

use crate::meta::Cover;
use crate::Err;

const MAX_EDGE: u32 = 1000;

pub(crate) fn is_jpeg(b: &[u8]) -> bool { b.len() >= 2 && b[0] == 0xFF && b[1] == 0xD8 }
pub(crate) fn is_png(b: &[u8]) -> bool { b.len() >= 8 && b[0..4] == [0x89, 0x50, 0x4E, 0x47] }

/// Validate and (when needed) downscale a cover image to fit MAX_EDGE.
pub(crate) fn prepare(raw: &[u8]) -> Result<Cover, Err> {
    if is_png(raw) {
        return Ok(Cover { data: raw.to_vec(), kind: 14 });
    }
    if !is_jpeg(raw) {
        return Err(Err::BadCover);
    }

    let mut decoder: Decoder<&[u8]> = Decoder::new(raw);
    let pixels: Vec<u8> = decoder.decode().map_err(|_| Err::BadCover)?;
    let info: ImageInfo = decoder.info().ok_or(Err::BadCover)?;
    let (w, h): (u32, u32) = (info.width as u32, info.height as u32);
    if w == 0 || h == 0 { return Err(Err::BadCover); }

    // already fits — the original bytes pass through untouched
    if w <= MAX_EDGE && h <= MAX_EDGE {
        return Ok(Cover { data: raw.to_vec(), kind: 13 });
    }

    let (nw, nh): (u32, u32) = fit(w, h, MAX_EDGE);
    match info.pixel_format {
        PixelFormat::L8 => {
            let scaled = resize_area(&pixels, w, h, 1, nw, nh);
            Ok(Cover { data: encode_jpeg(&scaled, nw, nh, ColorType::Luma)?, kind: 13 })
        }
        PixelFormat::RGB24 => {
            let scaled = resize_area(&pixels, w, h, 3, nw, nh);
            Ok(Cover { data: encode_jpeg(&scaled, nw, nh, ColorType::Rgb)?, kind: 13 })
        }
        _ => Err(Err::BadCover), // CMYK covers are beyond scope
    }
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

fn encode_jpeg(pixels: &[u8], w: u32, h: u32, color: ColorType) -> Result<Vec<u8>, Err> {
    let mut out: Vec<u8> = Vec::new();
    // q90 — visually lossless at 1000px; q100 bloats photographic covers ~3.5x
    let encoder: Encoder<&mut Vec<u8>> = Encoder::new(&mut out, 90);
    encoder
        .encode(pixels, w as u16, h as u16, color)
        .map_err(|_| Err::BadCover)?;
    Ok(out)
}
