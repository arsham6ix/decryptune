// Hand-rolled JPEG codec — zero external codec dependencies. Decode covers
// sequential + progressive baseline-Huffman JPEGs (SOF0/1/2, 8-bit, gray /
// YCbCr, restart markers, DC/AC successive approximation, EOBRUN); encode
// writes baseline SOF0 (gray or 4:4:4) with the standard Annex K tables.
// CMYK / lossless / arithmetic input surfaces as Err::BadCover.

mod decode;
mod encode;
#[cfg(test)]
mod tests;

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

// orthonormal 1-D DCT basis C(u)·cos((2x+1)uπ/16) — built at runtime because
// float methods are not const-evaluable; shared by IDCT (decode) and FDCT
// (encode), where the forward transform is the transposed use of this matrix
pub(super) fn build_cos() -> [[f32; 8]; 8] {
    let mut t: [[f32; 8]; 8] = [[0.0; 8]; 8];
    for (u, row) in t.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            let c: f32 = ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos();
            *cell = if u == 0 { 0.35355339 * c } else { 0.5 * c };
        }
    }
    t
}

pub(crate) use decode::decode;
pub(crate) use encode::encode;
