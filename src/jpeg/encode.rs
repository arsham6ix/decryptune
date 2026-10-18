// Baseline JPEG encoder (SOF0) — grayscale or 4:4:4 YCbCr, standard Annex K
// quantization/Huffman tables scaled libjpeg-style by quality. Emits SOI,
// APP0-JFIF, DQT, SOF0, DHT, SOS and pads the final entropy byte with 1-bits.

use super::ZIGZAG;
use super::build_cos;
use crate::Err;

const QUALITY: i32 = 90; // visually lossless at 1000px; q100 bloats ~3.5x

// Annex K quantization tables, natural (row-major) order
const Q_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
];
const Q_CHROMA: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
];

// Annex K Huffman specs — bits[l] counts codes of length l+1
const DC_VALS     : [u8; 12]  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DC_LUMA_BITS: [u8; 16]  = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const AC_LUMA_BITS: [u8; 16]  = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];
const AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
];
const AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
];

// canonical Huffman encoder table — symbol → (code, length)
struct EncTable {
    codes: [u16; 256],
    lens : [u8; 256]
}

impl EncTable {
    fn build(bits: &[u8; 16], vals: &[u8]) -> EncTable {
        let (mut codes, mut lens): ([u16; 256], [u8; 256]) = ([0; 256], [0; 256]);
        let (mut code, mut vi): (u32, usize) = (0, 0);
        for (li, count) in bits.iter().enumerate() {
            for _ in 0..*count {
                let s: usize = vals[vi] as usize;
                codes[s] = code as u16;
                lens[s] = li as u8 + 1;
                code += 1;
                vi += 1;
            }
            code <<= 1;
        }
        EncTable { codes, lens }
    }

    fn put(&self, bw: &mut BitWriter, sym: usize) {
        bw.put(self.codes[sym] as u32, self.lens[sym] as u32);
    }
}

// entropy byte writer — stuffs 0x00 after every 0xFF byte, pads the last
// partial byte with 1-bits so it can never read as a marker
struct BitWriter {
    out  : Vec<u8>,
    acc  : u64,
    nbits: u32
}

impl BitWriter {
    fn put(&mut self, code: u32, len: u32) {
        if len == 0 { return; }
        self.acc = (self.acc << len) | (code as u64 & ((1u64 << len) - 1));
        self.nbits += len;
        while self.nbits >= 8 {
            let b: u8 = (self.acc >> (self.nbits - 8)) as u8;
            self.out.push(b);
            if b == 0xFF { self.out.push(0x00); }
            self.nbits -= 8;
        }
        self.acc &= (1u64 << self.nbits) - 1;
    }

    fn flush(&mut self) {
        if self.nbits > 0 {
            let pad: u32 = 8 - self.nbits;
            self.put((1u32 << pad) - 1, pad);
        }
    }
}

// libjpeg quality scaling — q<50 uses 5000/q, else 200−2q
fn scale_qt(base: &[u16; 64]) -> [u16; 64] {
    let scale: i32 = if QUALITY < 50 { 5000 / QUALITY } else { 200 - 2 * QUALITY };
    let mut t: [u16; 64] = [0; 64];
    for i in 0..64 {
        t[i] = ((base[i] as i32 * scale + 50) / 100).clamp(1, 255) as u16;
    }
    t
}

// BT.601 YCbCr, full range (JFIF)
fn rgb_to_ycc(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n: usize = w * h;
    let (mut y, mut cb, mut cr): (Vec<u8>, Vec<u8>, Vec<u8>) = (vec![0; n], vec![0; n], vec![0; n]);
    for i in 0..n {
        let (r, g, b): (f32, f32, f32) = (rgb[i * 3] as f32, rgb[i * 3 + 1] as f32, rgb[i * 3 + 2] as f32);
        y[i]  = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        cb[i] = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0).round().clamp(0.0, 255.0) as u8;
        cr[i] = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0).round().clamp(0.0, 255.0) as u8;
    }
    (y, cb, cr)
}

// forward DCT via the shared orthonormal basis (F = C·f·Cᵀ) — row pass with
// u as horizontal frequency, column pass with v as vertical frequency, stored
// coef[v*8+u] to match the decoder's natural-order layout
fn fdct_block(px: &[u8; 64], cos: &[[f32; 8]; 8], out: &mut [i16; 64]) {
    let mut f: [[f32; 8]; 8] = [[0.0; 8]; 8];
    for p in 0..64 { f[p / 8][p % 8] = px[p] as f32 - 128.0; }
    let mut g: [[f32; 8]; 8] = [[0.0; 8]; 8];
    for y in 0..8 {
        for u in 0..8 {
            let mut s: f32 = 0.0;
            for x in 0..8 { s += f[y][x] * cos[u][x]; }
            g[y][u] = s;
        }
    }
    for v in 0..8 {
        for u in 0..8 {
            let mut s: f32 = 0.0;
            for y in 0..8 { s += g[y][u] * cos[v][y]; }
            out[v * 8 + u] = s.round() as i16;
        }
    }
}

fn quantize(coef: &mut [i16; 64], qt: &[u16; 64]) {
    for p in 0..64 {
        coef[p] = (coef[p] as f32 / qt[p] as f32).round() as i16;
    }
}

// 8×8 block with edge replication at the right/bottom borders
fn block8(plane: &[u8], w: usize, h: usize, bx: usize, by: usize) -> [u8; 64] {
    let mut b: [u8; 64] = [0; 64];
    for r in 0..8 {
        let y: usize = (by * 8 + r).min(h - 1);
        for c in 0..8 {
            let x: usize = (bx * 8 + c).min(w - 1);
            b[r * 8 + c] = plane[y * w + x];
        }
    }
    b
}

// amplitude category: bits needed for the signed value (0 when value is 0)
fn category(v: u32) -> u32 { 32 - v.leading_zeros() }

fn emit_block(bw: &mut BitWriter, coef: &[i16; 64], dc: &EncTable, ac: &EncTable, pred: &mut i32) {
    let diff: i32 = coef[0] as i32 - *pred;
    *pred = coef[0] as i32;
    let s: u32 = category(diff.unsigned_abs());
    dc.put(bw, s as usize);
    let amp: i32 = if diff < 0 { diff - 1 } else { diff }; // negative values offset by −1
    bw.put((amp & ((1 << s) - 1)) as u32, s);
    let mut run: usize = 0;
    for i in 1..64 {
        let v: i16 = coef[ZIGZAG[i]];
        if v == 0 { run += 1; continue; }
        while run > 15 { ac.put(bw, 0xF0); run -= 16; } // ZRL
        let s: u32 = category(u32::from(v.unsigned_abs()));
        ac.put(bw, (run << 4) | s as usize);
        let amp: i32 = if v < 0 { v as i32 - 1 } else { v as i32 };
        bw.put((amp & ((1 << s) - 1)) as u32, s);
        run = 0;
    }
    if run > 0 { ac.put(bw, 0x00); } // EOB
}

fn push_zigzag(out: &mut Vec<u8>, qt: &[u16; 64]) {
    for i in 0..64 { out.push(qt[ZIGZAG[i]] as u8); } // DQT stores zigzag order
}

fn push_dht(out: &mut Vec<u8>, class_id: u8, bits: &[u8; 16], vals: &[u8]) {
    out.extend_from_slice(&[0xFF, 0xC4]);
    out.extend_from_slice(&((2 + 1 + 16 + vals.len()) as u16).to_be_bytes());
    out.push(class_id);
    out.extend_from_slice(bits);
    out.extend_from_slice(vals);
}

/// Encode gray (ch=1) or RGB (ch=3) pixels into baseline JPEG bytes.
pub(crate) fn encode(pix: &[u8], w: usize, h: usize, ch: usize) -> Result<Vec<u8>, Err> {
    if w == 0 || h == 0 || (ch != 1 && ch != 3) || pix.len() < w * h * ch { return Err(Err::BadCover); }
    let (y, cb, cr): (Vec<u8>, Vec<u8>, Vec<u8>) = if ch == 1 {
        (pix.to_vec(), Vec::new(), Vec::new())
    } else {
        rgb_to_ycc(pix, w, h)
    };
    let (qt_y, qt_c): ([u16; 64], [u16; 64]) = (scale_qt(&Q_LUMA), scale_qt(&Q_CHROMA));
    let cos: [[f32; 8]; 8] = build_cos();
    let (dc_y, ac_y): (EncTable, EncTable) = (EncTable::build(&DC_LUMA_BITS, &DC_VALS), EncTable::build(&AC_LUMA_BITS, &AC_LUMA_VALS));
    let (dc_c, ac_c): (EncTable, EncTable) = (EncTable::build(&DC_CHROMA_BITS, &DC_VALS), EncTable::build(&AC_CHROMA_BITS, &AC_CHROMA_VALS));

    let mut out: Vec<u8> = Vec::with_capacity(w * h * ch / 2 + 1024);
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    out.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]); // APP0 JFIF 1.1
    if ch == 1 {
        out.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]); // DQT (luma only)
        push_zigzag(&mut out, &qt_y);
    } else {
        out.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x84, 0x00]); // DQT (luma + chroma)
        push_zigzag(&mut out, &qt_y);
        out.push(0x01);
        push_zigzag(&mut out, &qt_c);
    }
    out.extend_from_slice(&[0xFF, 0xC0]); // SOF0
    out.extend_from_slice(&(8 + ch as u16 * 3).to_be_bytes());
    out.push(8); // sample precision
    out.extend_from_slice(&(h as u16).to_be_bytes());
    out.extend_from_slice(&(w as u16).to_be_bytes());
    out.push(ch as u8);
    if ch == 1 {
        out.extend_from_slice(&[1, 0x11, 0]);
    } else {
        out.extend_from_slice(&[1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1]);
    }
    push_dht(&mut out, 0x00, &DC_LUMA_BITS, &DC_VALS);
    push_dht(&mut out, 0x10, &AC_LUMA_BITS, &AC_LUMA_VALS);
    if ch == 3 {
        push_dht(&mut out, 0x01, &DC_CHROMA_BITS, &DC_VALS);
        push_dht(&mut out, 0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALS);
    }
    out.extend_from_slice(&[0xFF, 0xDA]); // SOS
    out.extend_from_slice(&(6 + ch as u16 * 2).to_be_bytes());
    out.push(ch as u8);
    if ch == 1 {
        out.extend_from_slice(&[1, 0x00]);
    } else {
        out.extend_from_slice(&[1, 0x00, 2, 0x11, 3, 0x11]);
    }
    out.extend_from_slice(&[0, 63, 0]); // Ss, Se, Ah/Al

    // entropy data — MCU-interleaved: every unit writes Y, then Cb, then Cr
    let mut bw: BitWriter = BitWriter { out: Vec::new(), acc: 0, nbits: 0 };
    let (mut pred_y, mut pred_cb, mut pred_cr): (i32, i32, i32) = (0, 0, 0);
    for by in 0..h.div_ceil(8) {
        for bx in 0..w.div_ceil(8) {
            emit_unit(&y, w, h, bx, by, &qt_y, &dc_y, &ac_y, &mut bw, &mut pred_y, &cos);
            if ch == 3 {
                emit_unit(&cb, w, h, bx, by, &qt_c, &dc_c, &ac_c, &mut bw, &mut pred_cb, &cos);
                emit_unit(&cr, w, h, bx, by, &qt_c, &dc_c, &ac_c, &mut bw, &mut pred_cr, &cos);
            }
        }
    }
    bw.flush();
    out.append(&mut bw.out);
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn emit_unit(plane: &[u8], w: usize, h: usize, bx: usize, by: usize, qt: &[u16; 64], dc: &EncTable, ac: &EncTable,
             bw: &mut BitWriter, pred: &mut i32, cos: &[[f32; 8]; 8]) {
    let mut coef: [i16; 64] = [0; 64];
    fdct_block(&block8(plane, w, h, bx, by), cos, &mut coef);
    quantize(&mut coef, qt);
    emit_block(bw, &coef, dc, ac, pred);
}
