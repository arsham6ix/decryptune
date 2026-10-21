// Sequential + progressive JPEG decoder. Everything cover art needs — SOF0/1/2,
// 8-bit Huffman, gray/YCbCr, restart markers, DC/AC successive-approximation
// scans, EOBRUN — decodes to 1-channel gray or 3-channel RGB.

use super::ZIGZAG;
use crate::Err;

// hostile SOF dimensions can demand multi-GB coefficient buffers before any
// entropy byte is read — covers are downscaled to 1000px anyway, so 64MP
// (8192²) is far beyond anything legitimate; at the old 256MP cap a ~1MB
// hostile JPEG forced ~1GB of buffers and seconds of IDCT
const MAX_PIXELS: usize = 67_108_864; // 64 megapixels

struct HuffTable {
    counts    : [u8; 17],   // symbols per code length 1..=16
    first_code: [u16; 17],
    val_off   : [usize; 17],
    symbols   : Vec<u8>
}

impl HuffTable {
    // canonical code assignment with an overfull check: a table claiming more
    // codes than a length's code space holds would overflow the u16 code and
    // decode garbage downstream
    fn build(counts: &[u8; 17], symbols: Vec<u8>) -> Result<HuffTable, Err> {
        let (mut first_code, mut val_off, mut code, mut k): ([u16; 17], [usize; 17], u32, usize) = ([0; 17], [0; 17], 0, 0);
        for len in 1..=16 {
            first_code[len] = code as u16;
            val_off[len] = k;
            k += counts[len] as usize;
            code += counts[len] as u32;
            if code > (1u32 << len) { return Err(Err::BadCover); }
            code <<= 1;
        }
        Ok(HuffTable { counts: *counts, first_code, val_off, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, Err> {
        let mut code: i32 = 0;
        for len in 1..=16 {
            code = (code << 1) | br.bit()? as i32;
            let (fc, cnt): (i32, i32) = (self.first_code[len] as i32, self.counts[len] as i32);
            // cnt > 0 keeps an empty length from matching via a negative diff
            // (code below the canonical first code of a longer prefix)
            if cnt > 0 && code >= fc && code - fc < cnt {
                return Ok(self.symbols[self.val_off[len] + (code - fc) as usize]);
            }
        }
        Err(Err::BadCover)
    }
}

// entropy bit reader — stops (and 1-pads) at any marker; restart markers are
// consumed explicitly by the scan loop between restart intervals
struct BitReader<'a> {
    data : &'a [u8],
    pos  : usize,
    bits : u32,
    nbits: u32
}

impl<'a> BitReader<'a> {
    fn fill(&mut self) {
        while self.nbits <= 24 {
            let at_marker: bool = self.pos >= self.data.len()
                || (self.data[self.pos] == 0xFF && self.pos + 1 < self.data.len() && self.data[self.pos + 1] != 0x00);
            let b: u8 = if at_marker {
                0xFF // pad 1-bits — only reachable on truncated streams
            } else if self.data[self.pos] == 0xFF {
                self.pos += 2; // FF00 stuffing → literal FF
                0xFF
            } else {
                self.pos += 1;
                self.data[self.pos - 1]
            };
            self.bits = (self.bits << 8) | b as u32;
            self.nbits += 8;
        }
    }

    fn bit(&mut self) -> Result<u8, Err> {
        if self.nbits == 0 { self.fill(); }
        self.nbits -= 1;
        Ok((self.bits >> self.nbits & 1) as u8)
    }

    fn receive(&mut self, n: u32) -> Result<i32, Err> {
        if n == 0 { return Ok(0); }
        if self.nbits < n { self.fill(); }
        self.nbits -= n;
        Ok((self.bits >> self.nbits & ((1u32 << n) - 1)) as i32)
    }

    // drop buffered bits and consume the expected RSTn — the reader always
    // parks exactly on the marker, so the position is already byte-aligned
    fn take_restart(&mut self, idx: u8) -> Result<(), Err> {
        self.nbits = 0;
        if self.pos + 1 < self.data.len() && self.data[self.pos] == 0xFF && self.data[self.pos + 1] == 0xD0 + idx {
            self.pos += 2;
            Ok(())
        } else {
            Err(Err::BadCover)
        }
    }
}

// v < 2^(n-1) → negative extension
fn extend(v: i32, n: u32) -> i32 {
    if n > 0 && v < (1 << (n - 1)) { v - (1 << n) + 1 } else { v }
}

struct Component {
    id      : u8,
    h       : usize,
    v       : usize,
    tq      : usize,
    dc_sel  : usize,
    ac_sel  : usize,
    dc_pred : i32,
    blocks_x: usize,      // MCU-padded block grid (fixed by the frame, shared by all scans)
    blocks_y: usize,
    tile_x  : usize,      // frame tile bounds — non-interleaved scans stop here
    tile_y  : usize,
    coefs   : Vec<i16>,   // natural order, 64 per block
    plane   : Vec<u8>,    // filled by the final IDCT pass
    plane_w : usize,
    plane_h : usize
}

impl Component {
    fn block(&mut self, col: usize, row: usize) -> &mut [i16] {
        let b: usize = (row * self.blocks_x + col) * 64;
        &mut self.coefs[b..b + 64]
    }
}

// integer IDCT (zune/stb form, constants ×4096) — pass 1 transforms columns
// in place keeping 2 extra bits, pass 2 transforms rows into final pixels
const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

// one 1-D inverse transform; `final_pass` picks the output scaling:
// intermediate (>>10) or finished pixel (>>17, bias baked into SCALE_BITS)
fn idct_1d(v: &[i32; 8], final_pass: bool) -> [i32; 8] {
    // even part (frequencies 0, 2, 4, 6)
    let (e2, e6): (i32, i32) = (v[2], v[6]);
    let p1: i32 = (e2 + e6) * 2217;
    let t2: i32 = p1 + e6 * -7567;
    let t3: i32 = p1 + e2 * 3135;
    let (e0, e4): (i32, i32) = (v[0], v[4]);
    let t0: i32 = (e0 + e4) << 12;
    let t1: i32 = (e0 - e4) << 12;
    let bias: i32 = if final_pass { SCALE_BITS } else { 512 };
    let (x0, x3): (i32, i32) = (t0 + t3 + bias, t0 - t3 + bias);
    let (x1, x2): (i32, i32) = (t1 + t2 + bias, t1 - t2 + bias);
    // odd part (frequencies 7, 5, 3, 1)
    let (mut o0, mut o1, mut o2, mut o3): (i32, i32, i32, i32) = (v[7], v[5], v[3], v[1]);
    let p3: i32 = o0 + o2;
    let p4: i32 = o1 + o3;
    let p1: i32 = o0 + o3;
    let p2: i32 = o1 + o2;
    let p5: i32 = (p3 + p4) * 4816;
    o0 *= 1223;
    o1 *= 8410;
    o2 *= 12586;
    o3 *= 6149;
    let p1: i32 = p5 + p1 * -3685;
    let p2: i32 = p5 + p2 * -10497;
    let p3: i32 = p3 * -8034;
    let p4: i32 = p4 * -1597;
    o3 += p1 + p4;
    o2 += p2 + p3;
    o1 += p2 + p4;
    o0 += p1 + p3;
    let shift: i32 = if final_pass { 17 } else { 10 };
    [(x0 + o3) >> shift, (x1 + o2) >> shift, (x2 + o1) >> shift, (x3 + o0) >> shift,
     (x3 - o0) >> shift, (x2 - o1) >> shift, (x1 - o2) >> shift, (x0 - o3) >> shift]
}

fn idct_block(coef: &[i16], qt: &[u16; 64], out: &mut [u8; 64]) {
    let mut ws: [i32; 64] = [0; 64];
    for p in 0..64 { ws[p] = coef[p] as i32 * qt[p] as i32; } // dequantized
    for x in 0..8 {
        let mut v: [i32; 8] = [0; 8];
        for u in 0..8 { v[u] = ws[x + u * 8]; }
        let r: [i32; 8] = idct_1d(&v, false);
        for u in 0..8 { ws[x + u * 8] = r[u]; }
    }
    for y in 0..8 {
        let mut v: [i32; 8] = [0; 8];
        v.copy_from_slice(&ws[y * 8..y * 8 + 8]);
        let r: [i32; 8] = idct_1d(&v, true);
        for x in 0..8 {
            out[y * 8 + x] = r[x].clamp(0, 255) as u8;
        }
    }
}

// ---- scan decoders ----------------------------------------------------------

fn dc_first(comp: &mut Component, br: &mut BitReader<'_>, dc: &HuffTable, al: usize, col: usize, row: usize) -> Result<(), Err> {
    let t: u8 = dc.decode(br)?;
    comp.dc_pred += extend(br.receive(t as u32)?, t as u32);
    comp.block(col, row)[0] = (comp.dc_pred << al).clamp(-32768, 32767) as i16;
    Ok(())
}

fn dc_refine(comp: &mut Component, br: &mut BitReader<'_>, al: usize, col: usize, row: usize) -> Result<(), Err> {
    let b: &mut [i16] = comp.block(col, row);
    b[0] |= (br.bit()? as i16) << al;
    Ok(())
}

fn ac_first(coef: &mut [i16], br: &mut BitReader<'_>, ac: &HuffTable, ss: usize, se: usize, al: usize, eobrun: &mut u32) -> Result<(), Err> {
    if *eobrun > 0 { // inside an EOB run — this block reads nothing
        *eobrun -= 1;
        return Ok(());
    }
    let mut k: usize = ss;
    while k <= se {
        let rs: u8 = ac.decode(br)?;
        let (r, s): (usize, i32) = ((rs >> 4) as usize, (rs & 0xF) as i32);
        if s == 0 {
            if r == 15 { k += 16; continue; } // ZRL — next loop pass adds the 16th
            *eobrun = br.receive(r as u32)? as u32 + (1 << r) - 1; // current band consumed now
            break;
        }
        k += r;
        if k > se { return Err(Err::BadCover); }
        coef[ZIGZAG[k]] = (extend(br.receive(s as u32)?, s as u32) << al).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        k += 1;
    }
    Ok(())
}

// refine one existing coefficient (correction bit may bump ±1<<al)
fn refine_coef(coef: &mut i16, br: &mut BitReader<'_>, plus: i16) -> Result<(), Err> {
    if *coef != 0 && br.bit()? == 1 && *coef & plus == 0 {
        *coef += if *coef > 0 { plus } else { -plus };
    }
    Ok(())
}

fn ac_refine(comp_coef: &mut [i16], br: &mut BitReader<'_>, ac: &HuffTable, ss: usize, se: usize, al: usize, eobrun: &mut u32) -> Result<(), Err> {
    let plus: i16 = 1 << al;
    if *eobrun > 0 { // tail of a run — only existing coefficients get refined
        for k in ss..=se { refine_coef(&mut comp_coef[ZIGZAG[k]], br, plus)?; }
        *eobrun -= 1;
        return Ok(());
    }
    let mut k: usize = ss;
    while k <= se {
        let rs: u8 = ac.decode(br)?;
        let (mut r, s): (usize, i32) = ((rs >> 4) as usize, (rs & 0xF) as i32);
        if s != 0 {
            if s != 1 { return Err(Err::BadCover); } // refined magnitudes are fixed at ±1<<al
            let v: i16 = if br.bit()? == 1 { plus } else { -plus };
            // advance over nonzeros (correction bits) and r zeros, landing on
            // the target zero — libjpeg's `if (--r < 0) break`
            while k <= se {
                let p: usize = ZIGZAG[k];
                if comp_coef[p] != 0 {
                    refine_coef(&mut comp_coef[p], br, plus)?;
                } else {
                    if r == 0 { break; }
                    r -= 1;
                }
                k += 1;
            }
            if k > se { return Err(Err::BadCover); }
            comp_coef[ZIGZAG[k]] = v;
            k += 1;
        } else if r == 15 { // ZRL — advance over 15 zeros + the 16th, refining nonzeros
            while k <= se {
                let p: usize = ZIGZAG[k];
                if comp_coef[p] != 0 {
                    refine_coef(&mut comp_coef[p], br, plus)?;
                } else {
                    if r == 0 { break; }
                    r -= 1;
                }
                k += 1;
            }
            k += 1; // the outer for-loop's step past the 16th zero
        } else {
            // EOBn — run length 2^r+bits counts this block; its remaining band
            // still receives correction bits before the run continues
            *eobrun = br.receive(r as u32)? as u32 + (1 << r);
            for kk in k..=se { refine_coef(&mut comp_coef[ZIGZAG[kk]], br, plus)?; }
            *eobrun -= 1;
            break;
        }
    }
    Ok(())
}

// ---- marker parsing helpers -------------------------------------------------

fn read_be16(d: &[u8], pos: usize) -> Result<usize, Err> {
    if pos + 2 > d.len() { return Err(Err::BadCover); }
    Ok(u16::from_be_bytes([d[pos], d[pos + 1]]) as usize)
}

/// Decode a JPEG to (width, height, channels, pixels), channels ∈ {1, 3}.
pub(crate) fn decode(raw: &[u8]) -> Result<(usize, usize, usize, Vec<u8>), Err> {
    if raw.len() < 4 || raw[0..2] != [0xFF, 0xD8] { return Err(Err::BadCover); }
    let (mut pos, mut w, mut h): (usize, usize, usize) = (2, 0, 0);
    let (mut comps, mut qt): (Vec<Component>, [[u16; 64]; 4]) = (Vec::new(), [[0; 64]; 4]);
    let (mut dcs, mut acs): ([Option<HuffTable>; 4], [Option<HuffTable>; 4]) = ([None, None, None, None], [None, None, None, None]);
    let mut dri: usize = 0;
    let (mut hmax, mut vmax): (usize, usize) = (0, 0);
    let mut scans_done: usize = 0;

    'file: loop {
        // ---- header markers up to SOS ----
        let (scan, ss, se, ah, al): (Vec<usize>, usize, usize, usize, usize) = loop {
            if pos + 2 > raw.len() || raw[pos] != 0xFF { return Err(Err::BadCover); }
            let m: u8 = raw[pos + 1];
            pos += 2;
            match m {
                0xC0..=0xC2 => { // SOF0/1 sequential, SOF2 progressive
                    let seg: usize = read_be16(raw, pos)?;
                    if seg < 8 || pos + seg > raw.len() || raw[pos + 2] != 8 { return Err(Err::BadCover); }
                    h = u16::from_be_bytes([raw[pos + 3], raw[pos + 4]]) as usize;
                    w = u16::from_be_bytes([raw[pos + 5], raw[pos + 6]]) as usize;
                    let nc: usize = raw[pos + 7] as usize;
                    if w == 0 || h == 0 || w * h > MAX_PIXELS || (nc != 1 && nc != 3) || seg != 8 + nc * 3 { return Err(Err::BadCover); }
                    comps.clear();
                    for c in 0..nc {
                        let b: usize = pos + 8 + c * 3;
                        comps.push(Component {
                            id      : raw[b],
                            h       : (raw[b + 1] >> 4) as usize,
                            v       : (raw[b + 1] & 0xF) as usize,
                            tq      : (raw[b + 2] & 3) as usize,
                            dc_sel  : 0,
                            ac_sel  : 0,
                            dc_pred : 0,
                            blocks_x: 0,
                            blocks_y: 0,
                            tile_x  : 0,
                            tile_y  : 0,
                            coefs   : Vec::new(),
                            plane   : Vec::new(),
                            plane_w : 0,
                            plane_h : 0
                        });
                    }
                    hmax = comps.iter().map(|c| c.h).max().unwrap_or(0);
                    vmax = comps.iter().map(|c| c.v).max().unwrap_or(0);
                    if hmax == 0 || vmax == 0 || hmax > 4 || vmax > 4 { return Err(Err::BadCover); }
                    // block grid is padded to whole MCUs (interleaved scans write
                    // edge blocks of partial MCUs); non-interleaved scans stop at
                    // the frame tiles instead
                    let (mx, my): (usize, usize) = (w.div_ceil(hmax * 8), h.div_ceil(vmax * 8));
                    for c in &mut comps {
                        c.blocks_x = mx * c.h;
                        c.blocks_y = my * c.v;
                        c.tile_x = (w * c.h).div_ceil(hmax).div_ceil(8);
                        c.tile_y = (h * c.v).div_ceil(vmax).div_ceil(8);
                        c.coefs = vec![0i16; c.blocks_x * c.blocks_y * 64];
                    }
                    pos += seg;
                }
                0xC3 | 0xC5..=0xCF => return Err(Err::BadCover), // lossless / arithmetic
                0xC4 => { // DHT
                    let seg: usize = read_be16(raw, pos)?;
                    if pos + seg > raw.len() { return Err(Err::BadCover); }
                    let mut p: usize = pos + 2;
                    while p + 17 <= pos + seg {
                        let (tc, th): (usize, usize) = ((raw[p] >> 4) as usize, (raw[p] & 0xF) as usize);
                        if tc > 1 || th > 3 { return Err(Err::BadCover); }
                        let mut counts: [u8; 17] = [0; 17];
                        counts[1..].copy_from_slice(&raw[p + 1..p + 17]);
                        let total: usize = counts[1..].iter().map(|&c| c as usize).sum();
                        if p + 17 + total > pos + seg { return Err(Err::BadCover); }
                        // a DC symbol is a magnitude category (0..=15): anything
                        // larger would underflow the bit reader's bit count
                        if tc == 0 && raw[p + 17..p + 17 + total].iter().any(|&s| s > 15) {
                            return Err(Err::BadCover);
                        }
                        let table: HuffTable = HuffTable::build(&counts, raw[p + 17..p + 17 + total].to_vec())?;
                        if tc == 0 { dcs[th] = Some(table); } else { acs[th] = Some(table); }
                        p += 17 + total;
                    }
                    pos += seg;
                }
                0xDB => { // DQT
                    let seg: usize = read_be16(raw, pos)?;
                    if pos + seg > raw.len() { return Err(Err::BadCover); }
                    let mut p: usize = pos + 2;
                    while p < pos + seg {
                        let (pq, tq): (usize, usize) = ((raw[p] >> 4) as usize, (raw[p] & 0xF) as usize);
                        if pq > 1 || tq > 3 { return Err(Err::BadCover); }
                        for i in 0..64 {
                            qt[tq][ZIGZAG[i]] = if pq == 1 {
                                if p + 3 + i * 2 >= pos + seg { return Err(Err::BadCover); }
                                u16::from_be_bytes([raw[p + 2 + i * 2], raw[p + 3 + i * 2]])
                            } else {
                                if p + 1 + i >= pos + seg { return Err(Err::BadCover); }
                                raw[p + 1 + i] as u16
                            };
                        }
                        p += 1 + 64 * (pq + 1);
                    }
                    pos += seg;
                }
                0xDD => { // DRI
                    if read_be16(raw, pos)? != 4 { return Err(Err::BadCover); }
                    dri = read_be16(raw, pos + 2)?;
                    pos += 4;
                }
                0xDA => { // SOS
                    let seg: usize = read_be16(raw, pos)?;
                    if pos + seg > raw.len() { return Err(Err::BadCover); }
                    let ns: usize = raw[pos + 2] as usize;
                    if seg != 6 + ns * 2 || ns > comps.len() { return Err(Err::BadCover); }
                    let mut scan: Vec<usize> = Vec::with_capacity(ns);
                    for i in 0..ns {
                        let (id, sels): (u8, u8) = (raw[pos + 3 + i * 2], raw[pos + 4 + i * 2]);
                        let ci: usize = comps.iter().position(|c| c.id == id).ok_or(Err::BadCover)?;
                        comps[ci].dc_sel = (sels >> 4) as usize;
                        comps[ci].ac_sel = (sels & 0xF) as usize;
                        scan.push(ci);
                    }
                    let s: usize = pos + 3 + ns * 2;
                    let (ss, se, ah, al): (usize, usize, usize, usize) =
                        (raw[s] as usize, raw[s + 1] as usize, (raw[s + 2] >> 4) as usize, (raw[s + 2] & 0xF) as usize);
                    if ss > 63 || se > 63 || ah > 13 || al > 13 || ss > se { return Err(Err::BadCover); }
                    pos += seg;
                    break (scan, ss, se, ah, al);
                }
                0xD9 => { // EOI
                    if scans_done == 0 { return Err(Err::BadCover); }
                    break 'file;
                }
                _    => pos += read_be16(raw, pos)? // APPn/COM/… — skip by declared length
            }
        };

        pos = run_scan(raw, pos, &mut comps, &dcs, &acs, &scan, ss, se, ah, al, dri, w, h, hmax, vmax)?;
        scans_done += 1;
    }

    // ---- IDCT every block, then upsample and color-convert ----
    for c in &mut comps {
        c.plane_w = c.blocks_x * 8;
        c.plane_h = c.blocks_y * 8;
        c.plane = vec![0u8; c.plane_w * c.plane_h];
        for row in 0..c.blocks_y {
            for col in 0..c.blocks_x {
                let b: usize = (row * c.blocks_x + col) * 64;
                let mut px: [u8; 64] = [0; 64];
                idct_block(&c.coefs[b..b + 64], &qt[c.tq], &mut px);
                let d: usize = (row * 8) * c.plane_w + col * 8; // pixel row of the block's top edge
                for r in 0..8 {
                    c.plane[d + r * c.plane_w..d + r * c.plane_w + 8].copy_from_slice(&px[r * 8..r * 8 + 8]);
                }
            }
        }
    }

    if comps.len() == 1 {
        return Ok((w, h, 1, crop(&comps[0].plane, comps[0].plane_w, w, h)));
    }
    // JFIF frame order: Y, Cb, Cr (progressive scans cover comps one at a time)
    let y: Vec<u8> = upsample(&comps[0].plane, comps[0].plane_w, comps[0].plane_h, comps[0].h, comps[0].v, hmax, vmax, w, h);
    let cb: Vec<u8> = upsample(&comps[1].plane, comps[1].plane_w, comps[1].plane_h, comps[1].h, comps[1].v, hmax, vmax, w, h);
    let cr: Vec<u8> = upsample(&comps[2].plane, comps[2].plane_w, comps[2].plane_h, comps[2].h, comps[2].v, hmax, vmax, w, h);
    let mut rgb: Vec<u8> = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let (l, b, r): (f32, f32, f32) = (y[i] as f32, cb[i] as f32 - 128.0, cr[i] as f32 - 128.0);
        rgb[i * 3]     = (l + 1.402 * r).clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 1] = (l - 0.344136 * b - 0.714136 * r).clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 2] = (l + 1.772 * b).clamp(0.0, 255.0) as u8;
    }
    Ok((w, h, 3, rgb))
}

// one SOS payload: decode every block of every unit, honoring restart intervals
#[allow(clippy::too_many_arguments)]
fn run_scan(raw: &[u8], pos: usize, comps: &mut [Component], dcs: &[Option<HuffTable>; 4], acs: &[Option<HuffTable>; 4],
            scan: &[usize], ss: usize, se: usize, ah: usize, al: usize, dri: usize, w: usize, h: usize,
            hmax: usize, vmax: usize) -> Result<usize, Err> {
    let interleaved: bool = scan.len() > 1;
    let (ux_n, uy_n): (usize, usize) = if interleaved {
        (w.div_ceil(hmax * 8), h.div_ceil(vmax * 8))
    } else {
        (comps[scan[0]].tile_x, comps[scan[0]].tile_y)
    };
    for &ci in scan { comps[ci].dc_pred = 0; } // predictors reset at every scan
    let mut br: BitReader<'_> = BitReader { data: raw, pos, bits: 0, nbits: 0 };
    let (mut rst, mut unit, mut eobrun): (u8, usize, u32) = (0, 0, 0);
    for uy in 0..uy_n {
        for ux in 0..ux_n {
            if dri > 0 && unit > 0 && unit % dri == 0 {
                br.take_restart(rst)?;
                rst = (rst + 1) & 7;
                for &ci in scan { comps[ci].dc_pred = 0; }
            }
            for &ci in scan {
                let (bh, bv, dsel, asel): (usize, usize, usize, usize) =
                    (comps[ci].h, comps[ci].v, comps[ci].dc_sel, comps[ci].ac_sel);
                let blocks: Vec<(usize, usize)> = if interleaved {
                    (0..bv).flat_map(|by| (0..bh).map(move |bx| (ux * bh + bx, uy * bv + by))).collect()
                } else {
                    vec![(ux, uy)]
                };
                for (col, row) in blocks {
                    if ss == 0 && se == 0 { // DC-only scan
                        if ah == 0 {
                            let dc: &HuffTable = dcs[dsel].as_ref().ok_or(Err::BadCover)?;
                            dc_first(&mut comps[ci], &mut br, dc, al, col, row)?;
                        } else {
                            dc_refine(&mut comps[ci], &mut br, al, col, row)?;
                        }
                    } else if ah == 0 && ss == 0 && se == 63 { // full sequential scan
                        let dc: &HuffTable = dcs[dsel].as_ref().ok_or(Err::BadCover)?;
                        let ac: &HuffTable = acs[asel].as_ref().ok_or(Err::BadCover)?;
                        let t: u8 = dc.decode(&mut br)?;
                        let diff: i32 = extend(br.receive(t as u32)?, t as u32);
                        comps[ci].dc_pred += diff;
                        let pred: i16 = comps[ci].dc_pred as i16;
                        let coef: &mut [i16] = comps[ci].block(col, row);
                        coef[0] = pred;
                        let mut k: usize = 1;
                        while k < 64 {
                            let rs: u8 = ac.decode(&mut br)?;
                            let (r, s): (usize, i32) = ((rs >> 4) as usize, (rs & 0xF) as i32);
                            if s == 0 {
                                if r == 15 { k += 16; continue; }
                                break;
                            }
                            k += r;
                            if k > 63 { return Err(Err::BadCover); }
                            coef[ZIGZAG[k]] = extend(br.receive(s as u32)?, s as u32) as i16;
                            k += 1;
                        }
                    } else if ss > 0 { // progressive AC band
                        let ac: &HuffTable = acs[asel].as_ref().ok_or(Err::BadCover)?;
                        let coef: &mut [i16] = comps[ci].block(col, row);
                        if ah == 0 { ac_first(coef, &mut br, ac, ss, se, al, &mut eobrun)?; }
                        else { ac_refine(coef, &mut br, ac, ss, se, al, &mut eobrun)?; }
                    } else {
                        return Err(Err::BadCover); // partial DC band — not a legal scan
                    }
                }
            }
            unit += 1;
        }
    }
    Ok(br.pos) // next marker follows the entropy stream
}

// top-left w×h of a padded plane
fn crop(plane: &[u8], pw: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        out.extend_from_slice(&plane[y * pw..y * pw + w]);
    }
    out
}

// nearest-neighbor expansion for exotic sampling ratios
#[allow(clippy::too_many_arguments)]
fn upsample_plain(plane: &[u8], pw: usize, hs: usize, vs: usize, hmax: usize, vmax: usize, w: usize, oh: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; w * oh];
    for y in 0..oh {
        let sy: usize = y * vs / vmax;
        for x in 0..w {
            out[y * w + x] = plane[sy * pw + x * hs / hmax];
        }
    }
    out
}

// libjpeg "fancy" (triangle) upsampling — h2v1: horizontal 3:1 blend
fn upsample_h2v1(plane: &[u8], pw: usize, w: usize, oh: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; w * oh];
    for y in 0..oh {
        let row = &plane[y * pw..];
        let out_row = &mut out[y * w..y * w + w];
        for i in 0..w.div_ceil(2) {
            let cur: u32 = row[i.min(pw - 1)] as u32;
            let left: u32 = row[i.saturating_sub(1).min(pw - 1)] as u32;
            let right: u32 = row[(i + 1).min(pw - 1)] as u32;
            if i == 0 {
                out_row[0] = cur as u8;
                if w > 1 { out_row[1] = ((cur * 3 + right + 2) >> 2) as u8; }
            } else if i * 2 + 1 < w {
                out_row[i * 2] = ((cur * 3 + left + 1) >> 2) as u8;
                out_row[i * 2 + 1] = ((cur * 3 + right + 2) >> 2) as u8;
            } else {
                out_row[i * 2] = ((cur * 3 + left + 1) >> 2) as u8;
                if i * 2 + 1 < w { out_row[i * 2 + 1] = cur as u8; }
            }
        }
    }
    out
}

// libjpeg h2v2 fancy: vertical 3:1 blend into column sums, then horizontal 3:1
fn upsample_h2v2(plane: &[u8], pw: usize, ph: usize, w: usize, oh: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; w * oh];
    let cw: usize = w.div_ceil(2);
    for j in 0..oh.div_ceil(2) {
        let cur = &plane[j.min(ph - 1) * pw..];
        for dy in 0..2 {
            // an odd output height ends on a dy=0 row; the below-blend pass
            // would write row `oh` past the buffer, so it lands on `oh - 1`
            // (libjpeg's bottom-edge replication)
            let dst: usize = (2 * j + dy).min(oh - 1);
            let orow: usize = if dy == 0 { j.saturating_sub(1).min(ph - 1) } else { (j + 1).min(ph - 1) };
            let oth = &plane[orow * pw..];
            let orow_out = &mut out[dst * w..dst * w + w];
            let colsum = |i: usize| -> u32 { (cur[i.min(pw - 1)] as u32) * 3 + (oth[i.min(pw - 1)] as u32) };
            let (c0, c1): (u32, u32) = (colsum(0), colsum(1));
            orow_out[0] = ((c0 * 4 + 8) >> 4) as u8;
            if w > 1 { orow_out[1] = ((c0 * 3 + c1 + 7) >> 4) as u8; }
            for i in 1..cw {
                let (ci, cp, cn): (u32, u32, u32) = (colsum(i), colsum(i - 1), colsum(i + 1));
                if i * 2 < w { orow_out[i * 2] = ((ci * 3 + cp + 8) >> 4) as u8; }
                if i * 2 + 1 < w { orow_out[i * 2 + 1] = ((ci * 3 + cn + 7) >> 4) as u8; }
            }
            // libjpeg's last-column special cases land on the loop's final writes;
            // the clamped colsum(cw) replicates the edge like `* 4` / `+ cn` above
        }
    }
    out
}

// dispatch by sampling ratio
#[allow(clippy::too_many_arguments)]
fn upsample(plane: &[u8], pw: usize, ph: usize, hs: usize, vs: usize, hmax: usize, vmax: usize, w: usize, oh: usize) -> Vec<u8> {
    let (fx, fy): (usize, usize) = (hmax / hs, vmax / vs);
    match (fx, fy) {
        (2, 2) => upsample_h2v2(plane, pw, ph, w, oh),
        (2, 1) => upsample_h2v1(plane, pw, w, oh),
        _      => upsample_plain(plane, pw, hs, vs, hmax, vmax, w, oh),
    }
}
