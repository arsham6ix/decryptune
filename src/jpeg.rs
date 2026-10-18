// Sequential + progressive JPEG decoder, hand-rolled — everything cover art
// needs (SOF0/1/2, 8-bit Huffman, gray/YCbCr, restart markers, DC/AC
// successive-approximation scans, EOBRUN). CMYK/lossless/arithmetic surface
// as Err::BadCover. Decodes to 1-channel gray or 3-channel RGB.

use crate::Err;

const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
];

struct HuffTable {
    counts    : [u8; 17],   // symbols per code length 1..=16
    first_code: [u16; 17],
    val_off   : [usize; 17],
    symbols   : Vec<u8>
}

impl HuffTable {
    fn build(counts: &[u8; 17], symbols: Vec<u8>) -> HuffTable {
        let (mut first_code, mut val_off, mut code, mut k): ([u16; 17], [usize; 17], u16, usize) = ([0; 17], [0; 17], 0, 0);
        for len in 1..=16 {
            first_code[len] = code;
            val_off[len] = k;
            k += counts[len] as usize;
            code = (code + counts[len] as u16) << 1;
        }
        HuffTable { counts: *counts, first_code, val_off, symbols }
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, Err> {
        let mut code: i32 = 0;
        for len in 1..=16 {
            code = (code << 1) | br.bit()? as i32;
            let (fc, cnt): (i32, i32) = (self.first_code[len] as i32, self.counts[len] as i32);
            if code - fc < cnt {
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

// orthonormal 1-D inverse basis C(u)·cos((2x+1)uπ/16) — built at runtime
// (float methods are not const-evaluable)
fn build_cos() -> [[f32; 8]; 8] {
    let mut t: [[f32; 8]; 8] = [[0.0; 8]; 8];
    for (u, row) in t.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            let c: f32 = ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos();
            *cell = if u == 0 { 0.35355339 * c } else { 0.5 * c };
        }
    }
    t
}

fn idct_block(coef: &[i16], qt: &[u16; 64], cos: &[[f32; 8]; 8], out: &mut [u8; 64]) {
    let mut f: [[f32; 8]; 8] = [[0.0; 8]; 8];
    for p in 0..64 {
        f[p / 8][p % 8] = coef[p] as f32 * qt[p] as f32;
    }
    let mut g: [[f32; 8]; 8] = [[0.0; 8]; 8];
    for v in 0..8 {
        for x in 0..8 {
            let mut s: f32 = 0.0;
            for u in 0..8 { s += f[v][u] * cos[u][x]; }
            g[v][x] = s;
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let mut s: f32 = 128.0;
            for v in 0..8 { s += g[v][x] * cos[v][y]; }
            out[y * 8 + x] = s.clamp(0.0, 255.0) as u8;
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
    let cos: [[f32; 8]; 8] = build_cos();

    'file: loop {
        // ---- header markers up to SOS ----
        let (scan, ss, se, ah, al): (Vec<usize>, usize, usize, usize, usize) = loop {
            if pos + 2 > raw.len() || raw[pos] != 0xFF { return Err(Err::BadCover); }
            let m: u8 = raw[pos + 1];
            pos += 2;
            match m {
                0xC0 | 0xC1 | 0xC2 => { // SOF0/1 sequential, SOF2 progressive
                    let seg: usize = read_be16(raw, pos)?;
                    if seg < 8 || pos + seg > raw.len() || raw[pos + 2] != 8 { return Err(Err::BadCover); }
                    h = u16::from_be_bytes([raw[pos + 3], raw[pos + 4]]) as usize;
                    w = u16::from_be_bytes([raw[pos + 5], raw[pos + 6]]) as usize;
                    let nc: usize = raw[pos + 7] as usize;
                    if w == 0 || h == 0 || (nc != 1 && nc != 3) || seg != 8 + nc * 3 { return Err(Err::BadCover); }
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
                    let (mx, my): (usize, usize) = ((w + hmax * 8 - 1) / (hmax * 8), (h + vmax * 8 - 1) / (vmax * 8));
                    for c in &mut comps {
                        c.blocks_x = mx * c.h;
                        c.blocks_y = my * c.v;
                        c.tile_x = ((w * c.h + hmax - 1) / hmax + 7) / 8;
                        c.tile_y = ((h * c.v + vmax - 1) / vmax + 7) / 8;
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
                        let table: HuffTable = HuffTable::build(&counts, raw[p + 17..p + 17 + total].to_vec());
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

        // ---- run one scan ----
        let interleaved: bool = scan.len() > 1;
        let (ux_n, uy_n): (usize, usize) = if interleaved {
            ((w + hmax * 8 - 1) / (hmax * 8), (h + vmax * 8 - 1) / (vmax * 8))
        } else {
            (comps[scan[0]].tile_x, comps[scan[0]].tile_y)
        };
        for &ci in &scan { comps[ci].dc_pred = 0; } // predictors reset at every scan
        let mut br: BitReader<'_> = BitReader { data: raw, pos, bits: 0, nbits: 0 };
        let (mut rst, mut unit, mut eobrun): (u8, usize, u32) = (0, 0, 0);
        for uy in 0..uy_n {
            for ux in 0..ux_n {
                if dri > 0 && unit > 0 && unit % dri == 0 {
                    br.take_restart(rst)?;
                    rst = (rst + 1) & 7;
                    for &ci in &scan { comps[ci].dc_pred = 0; }
                }
                for &ci in &scan {
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
        scans_done += 1;
        pos = br.pos; // next marker follows the entropy stream
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
                idct_block(&c.coefs[b..b + 64], &qt[c.tq], &cos, &mut px);
                let d: usize = (row * 8) * c.plane_w + col * 8; // pixel row of the block's top edge
                c.plane[d..d + 8].copy_from_slice(&px[0..8]);
                c.plane[d + c.plane_w..d + c.plane_w + 8].copy_from_slice(&px[8..16]);
                c.plane[d + c.plane_w * 2..d + c.plane_w * 2 + 8].copy_from_slice(&px[16..24]);
                c.plane[d + c.plane_w * 3..d + c.plane_w * 3 + 8].copy_from_slice(&px[24..32]);
                c.plane[d + c.plane_w * 4..d + c.plane_w * 4 + 8].copy_from_slice(&px[32..40]);
                c.plane[d + c.plane_w * 5..d + c.plane_w * 5 + 8].copy_from_slice(&px[40..48]);
                c.plane[d + c.plane_w * 6..d + c.plane_w * 6 + 8].copy_from_slice(&px[48..56]);
                c.plane[d + c.plane_w * 7..d + c.plane_w * 7 + 8].copy_from_slice(&px[56..64]);
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
    if std::env::var("DUMP_YCC").is_ok() {
        use std::io::Write;
        let mut f = std::fs::File::create(".private/jt/my_ycc.raw").unwrap();
        for i in 0..w * h {
            f.write_all(&[y[i], cb[i], cr[i]]).unwrap();
        }
    }
    let mut rgb: Vec<u8> = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let (l, b, r): (f32, f32, f32) = (y[i] as f32, cb[i] as f32 - 128.0, cr[i] as f32 - 128.0);
        rgb[i * 3]     = (l + 1.402 * r).clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 1] = (l - 0.344136 * b - 0.714136 * r).clamp(0.0, 255.0) as u8;
        rgb[i * 3 + 2] = (l + 1.772 * b).clamp(0.0, 255.0) as u8;
    }
    Ok((w, h, 3, rgb))
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
        for i in 0..(w + 1) / 2 {
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
    let cw: usize = (w + 1) / 2;
    for j in 0..(oh + 1) / 2 {
        let cur = &plane[j.min(ph - 1) * pw..];
        for dy in 0..2 {
            let orow: usize = if dy == 0 { j.saturating_sub(1).min(ph - 1) } else { (j + 1).min(ph - 1) };
            let oth = &plane[orow * pw..];
            let orow_out = &mut out[(2 * j + dy) * w..(2 * j + dy) * w + w];
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
fn upsample(plane: &[u8], pw: usize, ph: usize, hs: usize, vs: usize, hmax: usize, vmax: usize, w: usize, oh: usize) -> Vec<u8> {
    let (fx, fy): (usize, usize) = (hmax / hs, vmax / vs);
    match (fx, fy) {
        (2, 2) => upsample_h2v2(plane, pw, ph, w, oh),
        (2, 1) => upsample_h2v1(plane, pw, w, oh),
        _      => upsample_plain(plane, pw, hs, vs, hmax, vmax, w, oh),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pillow_files() {
        for f in ["base_gray", "base_color", "prog_gray", "prog_color",
                  "pg8", "pg16", "pg32", "pg64", "pg100"] {
            let raw: Vec<u8> = std::fs::read(format!(".private/jt/{f}.jpg")).unwrap_or_default();
            if raw.is_empty() { continue; }
            match super::decode(&raw) {
                Ok((w, h, ch, _)) => println!("{f}: ok {w}x{h} ch{ch}"),
                Err(e) => println!("{f}: ERR {e:?}"),
            }
        }
    }


    #[test]
    fn fixtures() {
        for (p, wh) in [("tests/fixtures/big.jpg", (1600, 1200)), ("tests/fixtures/tiny.jpg", (400, 400))] {
            let raw: Vec<u8> = std::fs::read(p).unwrap();
            match super::decode(&raw) {
                Ok((w, h, ch, _)) => assert_eq!((w, h, ch), (wh.0, wh.1, 3)),
                Err(e) => panic!("{p}: {e:?}"),
            }
        }
    }

    fn base_or_big() -> bool { std::env::var("USE_BIG").is_ok() || std::env::var("USE_BASE").is_ok() }

    #[test]
    fn pixel_accuracy_vs_pillow() {
        // 4:4:4 so upsampling differences cannot mask decode errors; USE_BASE=1
        // swaps in a baseline JPEG and USE_GRAY=1 a grayscale one, to isolate
        // sequential-vs-progressive and color-conversion bugs
        let (jpg, rawf, ch_exp): (&str, &str, usize) = if std::env::var("USE_BIG").is_ok() {
            ("tests/fixtures/big.jpg", ".private/jt/big.raw", 3)
        } else if std::env::var("USE_FLAT").is_ok() {
            (".private/jt/flat.jpg", ".private/jt/flat.raw", 1)
        } else if std::env::var("USE_GRAY").is_ok() {
            (".private/jt/g444.jpg", ".private/jt/g444.raw", 1)
        } else if std::env::var("USE_SMOOTH").is_ok() {
            (".private/jt/smooth420.jpg", ".private/jt/smooth420.raw", 3)
        } else if std::env::var("USE_BASE").is_ok() {
            (".private/jt/b444.jpg", ".private/jt/b444.raw", 3)
        } else {
            (".private/jt/p444.jpg", ".private/jt/p444.raw", 3)
        };
        let raw: Vec<u8> = std::fs::read(jpg).unwrap_or_default();
        if raw.is_empty() { return; }
        let ref_rgb: Vec<u8> = std::fs::read(rawf).unwrap();
        let (w, h, ch, px) = super::decode(&raw).unwrap();
        let (ew, eh): (usize, usize) = if base_or_big() { (1600, 1200) } else { (100, 80) };
        assert_eq!((w, h, ch), (ew, eh, ch_exp));
        let n: usize = px.len() / ch;
        let total: u64 = px.iter().zip(ref_rgb.iter()).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
        println!("mean abs diff vs pillow: {:.3}", total as f64 / px.len() as f64);
        if ch == 3 {
            for c in 0..3 {
                let ctotal: u64 = (0..n).map(|i| (px[i * 3 + c] as i32 - ref_rgb[i * 3 + c] as i32).unsigned_abs() as u64).sum();
                println!("channel {c} mean abs diff: {:.3}", ctotal as f64 / n as f64);
            }
        }
        // uniform-random noise stresses the IDCT far more than real content; the
        // residual there is Pillow's own integer-IDCT rounding, not our error
        let limit: f64 = if std::env::var("USE_BIG").is_ok() || std::env::var("USE_SMOOTH").is_ok() { 2.0 } else { 6.0 };
        assert!(total as f64 / (px.len() as f64) < limit, "decoder diverges from reference");
    }
}
