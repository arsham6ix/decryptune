// Stable numeric error-code contract — messages are authored here and relayed
// by Python as "[code] + message" verbatim.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // DecryptFailed is part of the stable error-code contract
pub(crate) enum Err {
    InvalidKey          = 1,
    InvalidKid          = 2,
    InputNotFound       = 3,
    InvalidMp4          = 4,
    DecryptFailed       = 5,
    WriteFailed         = 6,
    KidMismatch         = 7,
    CorruptDetected     = 8,
    BadInput            = 9,
    BadOutput           = 10,
    MetaNeedsFragmented = 11,
    BadCover            = 12,
    CoverNotFound       = 13 // additive: a missing cover path must not read as a missing input
}

impl Err {
    pub(crate) fn msg(self) -> &'static str {
        match self {
            Err::InvalidKey          => "invalid key: must be 32 hex characters",
            Err::InvalidKid          => "invalid kid: must be 32 hex characters",
            Err::InputNotFound       => "input file not found",
            Err::CoverNotFound       => "cover file not found",
            Err::InvalidMp4          => "input is not a valid MP4",
            Err::DecryptFailed       => "decrypt failed: corrupt or unsupported fragment",
            Err::WriteFailed         => "failed to write output file",
            Err::KidMismatch         => "key does not match this file (KID mismatch)",
            Err::CorruptDetected     => "decrypted audio is corrupt (wrong or fake key)",
            Err::BadInput            => "invalid input: expected bytes, bytearray, memoryview, or a path",
            Err::BadOutput           => "invalid out: expected a path or None",
            Err::MetaNeedsFragmented => "metadata write requires a fragmented file (non-empty chunk offsets)",
            Err::BadCover            => "unsupported cover format: expected JPEG or PNG"
        }
    }
}

impl From<Err> for PyErr {
    fn from(e: Err) -> PyErr {
        PyException::new_err(format!("[{}] {}", e as i32, e.msg()))
    }
}
