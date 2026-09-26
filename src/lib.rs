mod boxes;
mod cover;
mod meta;
mod process;

use pyo3::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::types::{PyByteArray, PyBytes, PyString};

use std::sync::atomic::{AtomicU64, Ordering};

enum Input {
    Bytes(Vec<u8>),
    Path(String)
}

enum Output {
    Path(String),
    Return
}

impl Input {
    fn load(self) -> Result<Vec<u8>, Err> {
        match self {
            Input::Bytes(b) => Ok(b),
            Input::Path(p)  => std::fs::read(&p).map_err(|_| Err::InputNotFound)
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // DecryptFailed is part of the stable error-code contract
enum Err {
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
    fn msg(self) -> &'static str {
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

fn parse_hex16(hex: &str) -> Result<[u8; 16], Err> {
    let hex: &str = hex.trim();
    if hex.len() != 32 { return Err(Err::InvalidKey); }
    let mut key: [u8; 16] = [0u8; 16];
    for (i, b) in key.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|_| Err::InvalidKey)?;
    }
    Ok(key)
}

// key/KID must agree with the file's tenc — a wrong key would decrypt to silent noise
fn guard_kid(data: &[u8], kid: Option<&str>) -> Result<(), Err> {
    if let Some(kid) = kid {
        let expected: [u8; 16] = parse_hex16(kid).map_err(|_| Err::InvalidKid)?;
        if let Some(file_kid) = process::find_kid(data) {
            if file_kid != expected { return Err(Err::KidMismatch); }
        }
    }
    Ok(())
}

// unique temp name — two concurrent writers to the same output must not collide
// (counter + pid: uniqueness across processes too, not just within one)
fn tmp_path(out_path: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!("{}.{}.{}.tmp", out_path, std::process::id(), N.fetch_add(1, Ordering::Relaxed))
}

// write to <out>.<n>.tmp then rename — no partial outputs on crash
fn write_atomic(out_path: &str, data: &[u8]) -> Result<(), Err> {
    let tmp: String = tmp_path(out_path);
    std::fs::write(&tmp, data).map_err(|_| Err::WriteFailed)?;
    std::fs::rename(&tmp, out_path).map_err(|_| Err::WriteFailed)
}

fn parse_input(value: &Bound<'_, PyAny>) -> PyResult<Input> {
    Ok(if let Ok(b) = value.cast::<PyBytes>() {
        Input::Bytes(b.as_bytes().to_vec())
    } else if let Ok(b) = value.cast::<PyByteArray>() {
        Input::Bytes(b.to_vec())
    } else if let Ok(s) = value.cast::<PyString>() {
        Input::Path(s.to_str()?.to_string())
    } else if let Ok(p) = value.call_method0("__fspath__") { // os.PathLike
        Input::Path(p.cast::<PyString>()?.to_str()?.to_string())
    } else {
        // exotic buffer (memoryview, …) — abi3 has no PyBuffer; memoryview()
        // accepts only real buffer objects (bytes(123) would mean 123 zero bytes!)
        match value.py().import("builtins")?.getattr("memoryview")?.call1((value,)) {
            Ok(mv) => {
                let b = mv.call_method0("tobytes")?;
                Input::Bytes(b.cast::<PyBytes>()?.as_bytes().to_vec())
            }
            Err(_) => return Err(Err::BadInput.into())
        }
    })
}

fn parse_output(out: Option<&Bound<'_, PyAny>>) -> PyResult<Output> {
    match out {
        None => Ok(Output::Return),
        Some(v) => {
            // same path handling as input — os.PathLike accepted too
            let path: String = if let Ok(s) = v.cast::<PyString>() {
                s.to_str()?.to_string()
            } else if let Ok(p) = v.call_method0("__fspath__") {
                p.cast::<PyString>()?.to_str()?.to_string()
            } else {
                return Err(Err::BadOutput.into());
            };
            Ok(Output::Path(path))
        }
    }
}

// strict evidence check, shared by the sync and async tails
fn check_strict(processed: &process::Processed, key: Option<[u8; 16]>, strict: bool) -> PyResult<()> {
    if strict && key.is_some() {
        let bad_frags: Vec<String> = processed.fragments.iter()
            .filter(|f| f.corrupt == Some(true))
            .map(|f| format!("#{} chi2={:.0} ({}B)", f.index, f.chi2, f.bytes))
            .collect();
        if !bad_frags.is_empty() {
            return Err(PyException::new_err(format!(
                "[{}] decrypted audio is corrupt (wrong or fake key) — fragments: {}",
                Err::CorruptDetected as i32, bad_frags.join(", ")
            )));
        }
    }
    Ok(())
}

// sync tail: corrupt check + write-or-return
fn finish(processed: process::Processed, output: Output, key: Option<[u8; 16]>, strict: bool) -> PyResult<Option<Vec<u8>>> {
    check_strict(&processed, key, strict)?;
    match output {
        Output::Return  => Ok(Some(processed.data)),
        Output::Path(p) => {
            write_atomic(&p, &processed.data)?;
            Ok(None)
        }
    }
}

// shared head: load input, guard kid, run the fused pass
fn run(input: Input, key: Option<[u8; 16]>, kid: Option<String>, meta: Option<meta::CoreMeta>, strict: bool) -> Result<process::Processed, Err> {
    let data: Vec<u8> = input.load()?;
    if key.is_some() { guard_kid(&data, kid.as_deref())?; } // kid without key is meaningless
    process::process(&data, key.as_ref(), strict, meta.as_ref())
}

#[pyclass]
struct TuneMeta {
    #[pyo3(get, set)]
    title       : Option<String>,
    #[pyo3(get, set)]
    date        : Option<String>,
    #[pyo3(get, set)]
    artists     : Option<Vec<String>>,
    #[pyo3(get, set)]
    isrc        : Option<String>,
    #[pyo3(get, set)]
    track       : Option<u32>,
    #[pyo3(get, set)]
    track_total : Option<u32>,
    #[pyo3(get, set)]
    disc        : Option<u32>,
    #[pyo3(get, set)]
    disc_total  : Option<u32>,
    #[pyo3(get, set)]
    album       : Option<String>,
    #[pyo3(get, set)]
    genre       : Option<String>,
    #[pyo3(get, set)]
    lyrics      : Option<String>,
    #[pyo3(get, set)]
    composer    : Option<String>,
    #[pyo3(get, set)]
    cover       : Option<Py<PyAny>> // bytes or str path
}

#[pymethods]
impl TuneMeta {
    #[new]
    #[allow(clippy::too_many_arguments)] // 13 optional tag fields are the public API
    #[pyo3(signature = (title=None, date=None, artists=None, isrc=None,
                        track=None, track_total=None, disc=None, disc_total=None,
                        album=None, genre=None, lyrics=None, composer=None, cover=None))]
    fn new(
        title       : Option<String>,
        date        : Option<String>,
        artists     : Option<Vec<String>>,
        isrc        : Option<String>,
        track       : Option<u32>,
        track_total : Option<u32>,
        disc        : Option<u32>,
        disc_total  : Option<u32>,
        album       : Option<String>,
        genre       : Option<String>,
        lyrics      : Option<String>,
        composer    : Option<String>,
        cover       : Option<Py<PyAny>>
    ) -> Self {
        TuneMeta { title, date, artists, isrc, track, track_total, disc, disc_total,
                  album, genre, lyrics, composer, cover }
    }
}

// Python TuneMeta → Send CoreMeta: only raw byte copies and path strings happen under
// the GIL — cover disk read + JPEG prepare (decode/resize/encode) run in the worker
// None/0/empty fields mean "keep whatever the file already carries".
fn resolve_meta(py: Python<'_>, meta: Option<&TuneMeta>) -> PyResult<Option<meta::CoreMeta>> {
    let Some(m) = meta else { return Ok(None) };
    let cover: Option<meta::RawCover> = match &m.cover {
        None      => None,
        Some(obj) => {
            let bound = obj.bind(py);
            Some(if let Ok(b) = bound.cast::<PyBytes>() { meta::RawCover::Bytes(b.as_bytes().to_vec()) }
                else if let Ok(b) = bound.cast::<PyByteArray>() { meta::RawCover::Bytes(b.to_vec()) }
                else if let Ok(s) = bound.cast::<PyString>() { meta::RawCover::Path(s.to_str()?.to_string()) }
                else if let Ok(p) = bound.call_method0("__fspath__") { meta::RawCover::Path(p.cast::<PyString>()?.to_str()?.to_string()) }
                else {
                    match py.import("builtins")?.getattr("memoryview")?.call1((bound,)) {
                        Ok(mv) => {
                            let b = mv.call_method0("tobytes")?;
                            meta::RawCover::Bytes(b.cast::<PyBytes>()?.as_bytes().to_vec())
                        }
                        Err(_) => return Err(Err::BadCover.into())
                    }
                })
        }
    };
    let non_zero: fn(Option<u32>) -> Option<u32> = |v| v.filter(|n| *n > 0);
    let non_empty: fn(&Option<String>) -> Option<String> = |v| v.clone().filter(|s| !s.is_empty());
    let artists: Option<Vec<String>> = m.artists.clone().map(|a| a.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>()).filter(|a| !a.is_empty());

    Ok(Some(meta::CoreMeta {
        title       : non_empty(&m.title),
        date        : non_empty(&m.date),
        artists,
        isrc        : non_empty(&m.isrc),
        track       : non_zero(m.track),
        track_total : non_zero(m.track_total),
        disc        : non_zero(m.disc),
        disc_total  : non_zero(m.disc_total),
        album       : non_empty(&m.album),
        genre       : non_empty(&m.genre),
        lyrics      : non_empty(&m.lyrics),
        composer    : non_empty(&m.composer),
        cover
    }))
}

// Decrypt, fix, write metadata, optionally guard against fake keys — one pass.
// All parameters keyword-only; returns None (path out) or bytes (no out).
#[pyfunction]
#[pyo3(signature = (*, input, out=None, key=None, kid=None, meta=None, strict=false))]
fn proc<'py>(
    py     : Python<'py>,
    input  : &Bound<'py, PyAny>,
    out    : Option<&Bound<'py, PyAny>>,
    key    : Option<&str>,
    kid    : Option<&str>,
    meta   : Option<&TuneMeta>,
    strict : bool
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let input: Input = parse_input(input)?;
    let output: Output = parse_output(out)?;
    let key: Option<[u8; 16]> = key.map(parse_hex16).transpose()?;
    let core_meta: Option<meta::CoreMeta> = resolve_meta(py, meta)?;
    let kid: Option<String> = kid.map(|s| s.to_string());

    let result: Option<Vec<u8>> = py.detach(move || -> PyResult<Option<Vec<u8>>> {
        let processed: process::Processed = run(input, key, kid, core_meta, strict).map_err(PyErr::from)?;
        finish(processed, output, key, strict)
    })?;

    Ok(result.map(|bytes| PyBytes::new(py, &bytes).into_any()))
}

// async twin of proc: native coroutine, CPU work off-GIL, async file writes
#[pyfunction]
#[pyo3(signature = (*, input, out=None, key=None, kid=None, meta=None, strict=false))]
fn aproc<'py>(
    py     : Python<'py>,
    input  : &Bound<'py, PyAny>,
    out    : Option<&Bound<'py, PyAny>>,
    key    : Option<&str>,
    kid    : Option<&str>,
    meta   : Option<&TuneMeta>,
    strict : bool
) -> PyResult<Bound<'py, PyAny>> {
    let input: Input = parse_input(input)?;
    let output: Output = parse_output(out)?;
    let key: Option<[u8; 16]> = key.map(parse_hex16).transpose()?;
    let core_meta: Option<meta::CoreMeta> = resolve_meta(py, meta)?;
    let kid: Option<String> = kid.map(|s| s.to_string());

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let processed: process::Processed = tokio::task::spawn_blocking(move || run(input, key, kid, core_meta, strict))
            .await
            .map_err(|join| {
                // a worker panic must surface with its real message, not a fake write error
                let msg = join.try_into_panic().ok()
                    .and_then(|p| p.downcast_ref::<&str>().map(|s| s.to_string())
                        .or_else(|| p.downcast_ref::<String>().cloned()))
                    .unwrap_or_else(|| "internal worker panic".into());
                PyException::new_err(msg)
            })??;
        check_strict(&processed, key, strict)?;
        match output {
            // true async write — std::fs here would block a runtime worker
            Output::Path(p) => {
                let tmp: String = tmp_path(&p);
                tokio::fs::write(&tmp, &processed.data).await.map_err(|_| Err::WriteFailed)?;
                tokio::fs::rename(&tmp, &p).await.map_err(|_| Err::WriteFailed)?;
                Ok(None)
            }
            Output::Return  => Ok(Some(processed.data))
        }
    })
}

#[pymodule]
fn decryptune(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // runtime lives for the whole module lifetime — leak is intentional.
    // a second module load (importlib re-exec) must not panic: the first
    // runtime stays shared, the extra one is simply dropped
    let rt: tokio::runtime::Runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    let _ = pyo3_async_runtimes::tokio::init_with_runtime(Box::leak(Box::new(rt)));
    m.add_function(wrap_pyfunction!(proc, m)?)?;
    m.add_function(wrap_pyfunction!(aproc, m)?)?;
    m.add_class::<TuneMeta>()?;
    Ok(())
}
