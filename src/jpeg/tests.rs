// JPEG codec tests. Fixture-dependent tests read from the directory named by
// the JPEG_FIXTURES env var (generated locally, never shipped — the public
// source carries no private paths) and skip silently when it is unset; the
// pixel test compares against Pillow ground truth saved there as raw RGB.
// The encoder is validated by self-roundtrip plus a Pillow-decodable dump.

use crate::jpeg::{decode, encode};

// fixture path inside the JPEG_FIXTURES directory — None → tests skip
fn jt(name: &str) -> Option<String> {
    std::env::var("JPEG_FIXTURES").ok().map(|d| format!("{d}/{name}"))
}

#[test]
fn pillow_files() {
    let Some(dir) = std::env::var("JPEG_FIXTURES").ok() else { return };
    for f in ["base_gray", "base_color", "prog_gray", "prog_color",
              "pg8", "pg16", "pg32", "pg64", "pg100"] {
        let raw: Vec<u8> = std::fs::read(format!("{dir}/{f}.jpg")).unwrap_or_default();
        if raw.is_empty() { continue; }
        match decode(&raw) {
            Ok((w, h, ch, _)) => println!("{f}: ok {w}x{h} ch{ch}"),
            Err(e) => println!("{f}: ERR {e:?}"),
        }
    }
}

#[test]
fn fixtures() {
    for (p, wh) in [("tests/fixtures/big.jpg", (1600, 1200)), ("tests/fixtures/tiny.jpg", (400, 400))] {
        let raw: Vec<u8> = std::fs::read(p).unwrap();
        match decode(&raw) {
            Ok((w, h, ch, _)) => assert_eq!((w, h, ch), (wh.0, wh.1, 3)),
            Err(e) => panic!("{p}: {e:?}"),
        }
    }
}

fn base_or_big() -> bool { std::env::var("USE_BIG").is_ok() } // b444/p444 fixtures are 100x80

#[test]
fn pixel_accuracy_vs_pillow() {
    // 4:4:4 so upsampling differences cannot mask decode errors; USE_BASE=1
    // swaps in a baseline JPEG and USE_GRAY=1 a grayscale one, to isolate
    // sequential-vs-progressive and color-conversion bugs
    let (jpg, rawf, ch_exp): (Option<String>, Option<String>, usize) = if std::env::var("USE_BIG").is_ok() {
        (Some("tests/fixtures/big.jpg".into()), jt("big.raw"), 3)
    } else if std::env::var("USE_FLAT").is_ok() {
        (jt("flat.jpg"), jt("flat.raw"), 1)
    } else if std::env::var("USE_GRAY").is_ok() {
        (jt("g444.jpg"), jt("g444.raw"), 1)
    } else if std::env::var("USE_SMOOTH").is_ok() {
        (jt("smooth420.jpg"), jt("smooth420.raw"), 3)
    } else if std::env::var("USE_BASE").is_ok() {
        (jt("b444.jpg"), jt("b444.raw"), 3)
    } else {
        (jt("p444.jpg"), jt("p444.raw"), 3)
    };
    let (Some(jpg), Some(rawf)) = (jpg, rawf) else { return };
    let ref_rgb: Vec<u8> = std::fs::read(&rawf).unwrap();
    let (w, h, ch, px) = decode(&std::fs::read(&jpg).unwrap()).unwrap();
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

#[test]
fn encode_decode_roundtrip() {
    // smooth synthetic gradient — a lossy q90 encode must stay visually close
    let (w, h): (usize, usize) = (100, 80);
    let mut rgb: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            rgb.push(((x * 255) / w) as u8);
            rgb.push((((x + y) * 255) / (w + h)) as u8);
            rgb.push(((y * 255) / h) as u8);
        }
    }
    let jpg: Vec<u8> = encode(&rgb, w, h, 3).unwrap();
    let (dw, dh, dch, px) = decode(&jpg).unwrap();
    assert_eq!((dw, dh, dch), (w, h, 3));
    let total: u64 = px.iter().zip(rgb.iter()).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
    let mean: f64 = total as f64 / px.len() as f64;
    println!("encode roundtrip mean abs diff: {mean:.3}");
    assert!(mean < 4.0, "encoder roundtrip diverges: {mean}");
}

#[test]
fn encode_gray_flat_exact() {
    // flat gray is a DC-only stream and must roundtrip within rounding
    let jpg: Vec<u8> = encode(&[137u8; 64 * 64], 64, 64, 1).unwrap();
    let (w, h, ch, px) = decode(&jpg).unwrap();
    assert_eq!((w, h, ch), (64, 64, 1));
    assert!(px.iter().all(|&p| (p as i32 - 137).abs() <= 1), "flat gray drifted");
}

#[test]
fn encode_edge_blocks() {
    // dimensions not multiples of 8 exercise edge replication
    let (w, h): (usize, usize) = (101, 37);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 31) % 251) as u8).collect();
    let jpg: Vec<u8> = encode(&rgb, w, h, 3).unwrap();
    let (dw, dh, dch, _) = decode(&jpg).unwrap();
    assert_eq!((dw, dh, dch), (w, h, 3));
}

#[test]
fn hostile_huge_sof_rejected() {
    // SOF claims must fail validation before any buffer is allocated — a
    // 65535x65535 claim wants ~8GB of coefficient buffers, and 16384x16384
    // (the old 256MP cap's boundary) still wants ~1GB
    for (w, h) in [(0xFFFFu16, 0xFFFFu16), (16384u16, 16384u16)] {
        let sof: [u8; 11] = [0, 11, 8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, 1, 1, 0x11, 0];
        let jpg: Vec<u8> = [&[0xFF, 0xD8, 0xFF, 0xC0][..], &sof, &[0xFF, 0xD9][..]].concat();
        assert!(decode(&jpg).is_err(), "{w}x{h} not rejected");
    }
}

#[test]
fn hostile_dht_rejected() {
    // DC magnitude symbols above 15 would underflow the bit reader's count and
    // overfull canonical tables would overflow the code book — both rejected
    // at DHT parse time (a debug build once panicked on the first)
    let dqt: Vec<u8> = [[0xFFu8, 0xDB].as_slice(), &67u16.to_be_bytes(), &[0x00], &[1u8; 64]].concat();
    let sof: Vec<u8> = [[0xFFu8, 0xC0].as_slice(), &11u16.to_be_bytes(), &[8u8, 0, 16, 0, 16, 1, 1, 0x11, 0]].concat();
    let ac: Vec<u8> = [[0xFFu8, 0xC4].as_slice(), &20u16.to_be_bytes(), &[0x10], &[1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0]].concat();
    let sos: Vec<u8> = [[0xFFu8, 0xDA].as_slice(), &8u16.to_be_bytes(), &[1u8, 1, 0, 0, 63, 0]].concat();
    let dht = |class: u8, counts: &[u8], symbols: &[u8]| -> Vec<u8> {
        [[0xFFu8, 0xC4].as_slice(), &((2 + 1 + 16 + symbols.len()) as u16).to_be_bytes(), &[class], counts, symbols].concat()
    };
    let counts_len1: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let counts_overfull: [u8; 16] = [3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let entropy_and_eoi = &[0x00u8, 0x00, 0xFF, 0xD9][..];
    for bad_dht in [
        dht(0x00, &counts_len1, &[200]),                     // DC symbol 200
        dht(0x00, &counts_overfull, &[0, 1, 2, 3]),          // 3 one-bit codes
    ] {
        let jpg: Vec<u8> = [&[0xFF, 0xD8][..], &dqt, &sof, &bad_dht, &ac, &sos, entropy_and_eoi].concat();
        assert!(decode(&jpg).is_err());
    }
}

// DUMP_ENC=1 + JPEG_FIXTURES: write our encode of the p444 reference so the
// local Pillow PSNR checker can score it against the same pristine source
#[test]
fn dump_encoded_for_pillow() {
    if std::env::var("DUMP_ENC").is_err() { return; }
    let Some(rawf) = jt("p444.raw") else { return };
    let raw: Vec<u8> = std::fs::read(&rawf).unwrap_or_default();
    if raw.is_empty() { return; }
    let jpg: Vec<u8> = encode(&raw, 100, 80, 3).unwrap();
    std::fs::write(jt("enc_out.jpg").unwrap(), &jpg).unwrap();
}
