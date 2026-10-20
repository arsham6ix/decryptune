// Input/output plumbing: bytes | bytearray | memoryview | path dispatch and
// the zero-copy borrow of immutable bytes across the GIL release.

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyString};

use crate::error::Err;

pub(crate) enum Input {
    Owned(Vec<u8>),
    // PyBytes is immutable, so borrowing its buffer across a GIL release is
    // sound: the Py<> owner pins the object alive and nothing can change it
    // (the owner field is intentionally never read — it only anchors the slice)
    Borrowed(#[allow(dead_code)] Py<PyBytes>, &'static [u8]),
    Path(String)
}

pub(crate) enum InputData {
    // the owner is never read — it only anchors the borrowed slice
    Owned(Vec<u8>),
    Borrowed(#[allow(dead_code)] Py<PyBytes>, &'static [u8])
}

impl InputData {
    pub(crate) fn bytes(&self) -> &[u8] {
        match self { InputData::Owned(v) => v, InputData::Borrowed(_, s) => s }
    }
}

impl Input {
    pub(crate) fn load(self) -> Result<InputData, Err> {
        Ok(match self {
            Input::Owned(v)       => InputData::Owned(v),
            Input::Borrowed(o, s) => InputData::Borrowed(o, s),
            Input::Path(p)        => InputData::Owned(std::fs::read(&p).map_err(|_| Err::InputNotFound)?)
        })
    }
}

pub(crate) enum Output {
    Path(String),
    Return
}

// the shared bytes/path dispatch: bytes | bytearray | str path | __fspath__ | memoryview
pub(crate) enum Extracted {
    Bytes(Vec<u8>),
    Borrowed(Py<PyBytes>, &'static [u8]),
    Path(String)
}

pub(crate) fn extract_bytes_or_path(value: &Bound<'_, PyAny>, bad: Err) -> PyResult<Extracted> {
    Ok(if let Ok(b) = value.cast::<PyBytes>() {
        // immutable bytes — pin the object and borrow its buffer instead of
        // copying; from_raw_parts re-lifetimes the slice, sound because the
        // Py<> owner stored beside it outlives every use of the slice
        let owner: Py<PyBytes> = b.clone().unbind();
        let s: &[u8] = b.as_bytes();
        Extracted::Borrowed(owner, unsafe { std::slice::from_raw_parts(s.as_ptr(), s.len()) })
    } else if let Ok(b) = value.cast::<PyByteArray>() {
        Extracted::Bytes(b.to_vec()) // mutable — a copy is mandatory
    } else if let Ok(s) = value.cast::<PyString>() {
        // to_str fails only on lone surrogates (real on Windows via os.fsdecode) —
        // the Rust core takes String paths only, so that input has no valid encoding
        Extracted::Path(s.to_str().map_err(|_| bad)?.to_string())
    } else if let Ok(p) = value.call_method0("__fspath__") { // os.PathLike
        // fspath may legally return bytes — map that here, never a raw downcast error
        Extracted::Path(if let Ok(s) = p.cast::<PyString>() {
            s.to_str().map_err(|_| bad)?.to_string()
        } else if let Ok(b) = p.cast::<PyBytes>() {
            String::from_utf8_lossy(b.as_bytes()).into_owned()
        } else {
            return Err(bad.into());
        })
    } else {
        // exotic buffer (memoryview, …) — abi3 has no PyBuffer; memoryview()
        // accepts only real buffer objects (bytes(123) would mean 123 zero bytes!)
        match value.py().import("builtins")?.getattr("memoryview")?.call1((value,)) {
            Ok(mv) => {
                let b = mv.call_method0("tobytes")?;
                Extracted::Bytes(b.cast::<PyBytes>()?.as_bytes().to_vec())
            }
            Err(_) => return Err(bad.into()),
        }
    })
}

pub(crate) fn parse_input(value: &Bound<'_, PyAny>) -> PyResult<Input> {
    Ok(match extract_bytes_or_path(value, Err::BadInput)? {
        Extracted::Bytes(b)          => Input::Owned(b),
        Extracted::Borrowed(o, s)    => Input::Borrowed(o, s),
        Extracted::Path(p)           => Input::Path(p)
    })
}

pub(crate) fn parse_output(out: Option<&Bound<'_, PyAny>>) -> PyResult<Output> {
    match out {
        None => Ok(Output::Return),
        Some(v) => {
            let path: String = if let Ok(s) = v.cast::<PyString>() {
                // lone surrogates have no Rust-side String encoding — coded error, not a leak
                s.to_str().map_err(|_| Err::BadOutput)?.to_string()
            } else if let Ok(p) = v.call_method0("__fspath__") { // os.PathLike
                // same fspath-bytes rule as input — a coded error, not a raw downcast
                if let Ok(s) = p.cast::<PyString>() {
                    s.to_str().map_err(|_| Err::BadOutput)?.to_string()
                } else if let Ok(b) = p.cast::<PyBytes>() {
                    String::from_utf8_lossy(b.as_bytes()).into_owned()
                } else {
                    return Err(Err::BadOutput.into());
                }
            } else {
                return Err(Err::BadOutput.into());
            };
            Ok(Output::Path(path))
        }
    }
}
