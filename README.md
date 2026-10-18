# decryptune

Fast, native M4A (fMP4/CENC) decryptor + MP4 sanitizer + iTunes tagger in one lightweight, self-contained call — a Rust core with a sync/async Python API that replaces the usual chain of external tools (no subprocesses, no ffmpeg, no dependencies) and runs fully GIL-free, so async callers can fan out without blocking the event loop.

## Install

```bash
pip install decryptune
```

From source (needs Rust 1.85+ and Python 3.12+): `pip install maturin && maturin build --release` — this is also the macOS path (it builds and runs the same; prebuilt wheels ship for Linux and Windows).

## Usage

```python
from decryptune import DecrypTune, TuneMeta

sp = DecrypTune()

sp.proc(
    input="track.m4a",           # bytes / bytearray / memoryview / path
    out="out.m4a",               # destination path; None → the processed bytes are returned
    key="<32 hex>",              # None → fix-only pass (headers sanitized, payload untouched)
    kid="<32 hex>",              # verified against the file's tenc; ignored without key
    meta=TuneMeta(
        title="Song",
        artists=["A"],
        isrc="...",
        cover="cover.jpg",       # bytes or path — JPEG/PNG; oversized JPEGs shrink to ≤1000px
        # date=..., album=..., genre=..., lyrics=..., composer=...,
        # track=..., track_total=..., disc=..., disc_total=...
    ),
    strict=True                  # raise (code 8, per-fragment chi² evidence) when a fragment
)                                # decrypts to statistical noise — a wrong/fake key

# async twin — same options, same result:
out_bytes = await sp.aproc(...)
```

Rules that hold for both `proc` and `aproc`:

- Every parameter is **keyword-only**.
- `out` is a path → the file is written **atomically** (tmp + rename) and `None` is returned; `out` is `None` → the processed `bytes` are returned.
- `key=None` → **fix-only pass**: headers sanitized, audio payload untouched.
- A `meta` **wipes** the existing tag block and rebuilds it solely from the given fields — `None` / `0` / empty values are skipped; without `meta` the original metadata passes through untouched.
- Any problem raises `DecrypTuneError` with a stable numeric code (1–13) and a message authored in the native core: `1` invalid key · `2` invalid kid · `3` input not found · `4` not an MP4 · `6` write failed · `7` KID mismatch · `8` corrupt audio detected · `9` bad input type · `10` bad out type · `11` tags need a fragmented file · `12` bad cover · `13` cover file not found.

## Benchmark

Native Windows 11 (windows-x86_64 wheel), Intel i7-11800H (8C/16T), Python 3.13, real 3.0MB fragmented track:

| scenario | result |
|---|---|
| decrypt + fix, strict (bytes in/out) | 3.0 ms median — ~1.0 GB/s |
| fix-only pass (no key) | 1.0 ms |
| full tags + cover resize (877KB JPEG) | 219 ms |
| 32 calls: sequential → `asyncio.gather` (bytes) | 97 ms → 30 ms (**×3.3**) |
| 32 calls with path out (real writes) | 46 ms (**×2.1**) |
| event-loop max lag under 8 concurrent heavy calls | < 15 ms (Windows timer granularity) |

Fragment work (decrypt + strict evidence check) spreads across cores on large files: a 128MB fragmented file runs decrypt+strict in ~0.1s.

## License

MIT — see [LICENSE](LICENSE).
