mod boxes;
mod cover;
mod error;
mod input;
mod jpeg;
mod meta;
mod process;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use error::Err;
use input::{parse_input, parse_output, InputData, Output};

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

// write to <out>.<n>.tmp then rename — no partial outputs on crash.
// create_new (O_CREAT|O_EXCL) refuses a symlink pre-placed at the predictable
// tmp name and fails on collision instead of clobbering an existing file
fn write_atomic(out_path: &str, data: &[u8]) -> Result<(), Err> {
    let tmp: String = tmp_path(out_path);
    let mut file: std::fs::File = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp)
        .map_err(|_| Err::WriteFailed)?;
    if file.write_all(data).is_err() || std::fs::rename(&tmp, out_path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return Err(Err::WriteFailed);
    }
    Ok(())
}

// strict evidence check, shared by the sync and async tails; `guarded` is
// precomputed (strict && key given) because the key copy dies in its worker
fn check_strict(fragments: &[process::FragmentStat], guarded: bool) -> PyResult<()> {
    if guarded {
        let bad_frags: Vec<String> = fragments.iter()
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

// best-effort key hygiene: volatile writes defeat dead-store elimination so
// the worker's AES key copy is cleared instead of lingering in freed memory
fn wipe_key(key: &mut Option<[u8; 16]>) {
    if let Some(k) = key.as_mut() {
        for b in k.iter_mut() { unsafe { std::ptr::write_volatile(b, 0u8); } }
    }
}

// pass 1, shared by proc/aproc: load input, guard kid, measure the exact
// output length — all off the GIL; the loaded data and the metadata must
// survive to pass 2, so the metadata comes back out. Only `has_key` crosses
// here — the key itself never needs to reach the measure phase
fn load_and_measure(input: input::Input, has_key: bool, kid: Option<String>, meta: Option<meta::CoreMeta>)
    -> Result<(InputData, usize, Option<meta::CoreMeta>), Err> {
    let d: InputData = input.load()?;
    if has_key { guard_kid(d.bytes(), kid.as_deref())?; }
    let mut meta = meta;
    if let Some(m) = meta.as_mut() { meta::pre_resolve_cover(m)?; } // cover codec work once
    let len: usize = process::measure(d.bytes(), meta.as_ref())?;
    Ok((d, len, meta))
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

// parsed proc/aproc arguments — every Send piece is resolved under the GIL
// once, then moved wholesale into the worker
struct Args {
    input    : input::Input,
    output   : Output,
    key      : Option<[u8; 16]>,
    kid      : Option<String>,
    core_meta: Option<meta::CoreMeta>
}

fn parse_args(
    py   : Python<'_>,
    input: &Bound<'_, PyAny>,
    out  : Option<&Bound<'_, PyAny>>,
    key  : Option<&str>,
    kid  : Option<&str>,
    meta : Option<&TuneMeta>
) -> PyResult<Args> {
    Ok(Args {
        input    : parse_input(input)?,
        output   : parse_output(out)?,
        key      : key.map(parse_hex16).transpose()?,
        kid      : kid.map(|s| s.to_string()),
        core_meta: resolve_meta(py, meta)?
    })
}

// Python TuneMeta → Send CoreMeta: only raw byte copies and path strings happen under
// the GIL — cover disk read + JPEG prepare (decode/resize/encode) run in the worker
// None/0/empty fields mean "keep whatever the file already carries".
fn resolve_meta(py: Python<'_>, meta: Option<&TuneMeta>) -> PyResult<Option<meta::CoreMeta>> {
    let Some(m) = meta else { return Ok(None) };
    let cover: Option<meta::RawCover> = match &m.cover {
        None      => None,
        Some(obj) => Some(match input::extract_bytes_or_path(obj.bind(py), Err::BadCover)? {
            input::Extracted::Bytes(b)       => meta::RawCover::Bytes(b),
            // covers are small (≤1MB) — a copy here keeps RawCover simple
            input::Extracted::Borrowed(_, s) => meta::RawCover::Bytes(s.to_vec()),
            input::Extracted::Path(p)        => meta::RawCover::Path(p)
        })
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
        cover,
        cover_ready : None
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
    let Args { input, output, key, kid, core_meta }: Args = parse_args(py, input, out, key, kid, meta)?;
    let (data, len, core_meta): (InputData, usize, Option<meta::CoreMeta>) =
        py.detach(move || load_and_measure(input, key.is_some(), kid, core_meta)).map_err(PyErr::from)?;
    let guarded: bool = strict && key.is_some();

    match output {
        Output::Path(p) => {
            py.detach(move || -> PyResult<()> {
                let mut key: Option<[u8; 16]> = key;
                let mut buf: Vec<u8> = vec![0u8; len]; // exact-size output, no over-allocation
                let fragments: Vec<process::FragmentStat> =
                    process::build(data.bytes(), key.as_ref(), strict, core_meta.as_ref(), &mut buf)?;
                wipe_key(&mut key);
                check_strict(&fragments, guarded)?;
                write_atomic(&p, &buf)?;
                Ok(())
            })?;
            Ok(None)
        }
        Output::Return => {
            // build straight into the freshly allocated Python bytes — no
            // intermediate copy; the object is exclusively owned until the
            // handle is returned, so the detached build's &mut is exclusive
            let bytes: Bound<'py, PyBytes> = PyBytes::new_with(py, len, |_buf: &mut [u8]| Ok(()))?;
            // the buffer base crosses the detach as a usize — a captured raw
            // pointer would break Send (disjoint capture hoists it out)
            let base: usize = bytes.as_bytes().as_ptr() as usize;
            let fragments: Vec<process::FragmentStat> = py.detach(move || -> PyResult<Vec<process::FragmentStat>> {
                let mut key: Option<[u8; 16]> = key;
                let buf: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, len) };
                let fragments: Vec<process::FragmentStat> =
                    process::build(data.bytes(), key.as_ref(), strict, core_meta.as_ref(), buf)?;
                wipe_key(&mut key);
                Ok(fragments)
            })?;
            check_strict(&fragments, guarded)?;
            Ok(Some(bytes.into_any()))
        }
    }
}

// a worker panic must surface with its real message, not a fake error
fn join_err(join: tokio::task::JoinError) -> PyErr {
    let msg: String = join.try_into_panic().ok()
        .and_then(|p| p.downcast_ref::<&str>().map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned()))
        .unwrap_or_else(|| "internal worker panic".into());
    PyException::new_err(msg)
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
    let Args { input, output, key, kid, core_meta }: Args = parse_args(py, input, out, key, kid, meta)?;
    let guarded: bool = strict && key.is_some();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (data, len, core_meta): (InputData, usize, Option<meta::CoreMeta>) = tokio::task::spawn_blocking(
            move || load_and_measure(input, key.is_some(), kid, core_meta),
        ).await.map_err(join_err)??;
        match output {
            Output::Path(p) => {
                // write+rename run inside ONE blocking task with the build — a
                // cancelled future cannot split the pair and leave a tmp behind
                tokio::task::spawn_blocking(move || -> PyResult<()> {
                    let mut key: Option<[u8; 16]> = key;
                    let mut buf: Vec<u8> = vec![0u8; len];
                    let fragments: Vec<process::FragmentStat> =
                        process::build(data.bytes(), key.as_ref(), strict, core_meta.as_ref(), &mut buf)?;
                    wipe_key(&mut key);
                    check_strict(&fragments, guarded)?;
                    write_atomic(&p, &buf)?;
                    Ok(())
                }).await.map_err(join_err)??;
                Ok(None)
            }
            Output::Return => {
                // allocate the result under a brief GIL attach, then build off-GIL
                // straight into its buffer — no intermediate copy. The PyBytes
                // owner moves INTO the build task: a cancelled future drops its
                // own references but the buffer stays referenced for the whole
                // write (dropping it early would free the object under the writer)
                let (base, result): (usize, Py<PyBytes>) = Python::attach(|py| -> PyResult<(usize, Py<PyBytes>)> {
                    let b: Bound<'_, PyBytes> = PyBytes::new_with(py, len, |_buf: &mut [u8]| Ok(()))?;
                    Ok((b.as_bytes().as_ptr() as usize, b.unbind()))
                })?;
                let (fragments, result): (Vec<process::FragmentStat>, Py<PyBytes>) =
                    tokio::task::spawn_blocking(move || -> PyResult<(Vec<process::FragmentStat>, Py<PyBytes>)> {
                        let mut key: Option<[u8; 16]> = key;
                        let buf: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, len) };
                        let fragments: Vec<process::FragmentStat> =
                            process::build(data.bytes(), key.as_ref(), strict, core_meta.as_ref(), buf)?;
                        wipe_key(&mut key);
                        Ok((fragments, result))
                    }).await.map_err(join_err)??;
                check_strict(&fragments, guarded)?;
                Ok(Some(result))
            }
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
