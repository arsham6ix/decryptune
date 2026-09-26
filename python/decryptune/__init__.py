"""decryptune — fast, native M4A/CENC decryptor + MP4 sanitizer + iTunes tagger.

Wheel distribution: the native extension ships INSIDE this package and is
fixed at install time — no paths, nothing to swap. For the standalone
workflow (extension file + path) use the decryptune.py wrapper from the source
tree. Keep this file in sync with it (same API surface, loader differs).
"""

from . import decryptune as _native

__version__ = "1.0.0"
__all__ = ["DecrypTune", "DecrypTuneError", "TuneMeta", "__version__"]


class DecrypTuneError(Exception):
    """Processing error raised by the native core.

    Attributes:
        code: stable numeric error code (defined in the Rust core)
        message: full human-readable message, authored by the native core
    """

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

    def __repr__(self) -> str:
        return f"DecrypTuneError(code={self.code}, message={self.message!r})"


# all error messages are authored by the native core — this wrapper relays [code] + message
def _to_err(exc: Exception) -> DecrypTuneError:
    msg: str = str(exc)
    end: int = msg.find("]")
    if msg.startswith("[") and end != -1:
        try:
            return DecrypTuneError(int(msg[1:end]), msg[end + 1:].strip())
        except ValueError:
            pass
    return DecrypTuneError(0, msg)


class DecrypTune:
    """Interface to the decryptune native extension bundled in this wheel.

    The native module is fixed at install time — DecrypTune takes no path.
    """

    __slots__ = ("_lib",)

    def __init__(self) -> None:
        self._lib = _native

    def __repr__(self) -> str:
        return f"DecrypTune(lib={self._lib.__file__!r})"

    def proc(
        self,
        *,
        input: bytes | bytearray | memoryview | str,
        out: str | None = None,
        key: str | None = None,
        kid: str | None = None,
        meta: "TuneMeta | None" = None,
        strict: bool = False
    ) -> bytes | None:
        """Decrypt, fix, write metadata, and optionally guard — one pass.

        - input: file content as bytes/bytearray/memoryview (no disk read) or
                 a path (read from disk)
        - out:   destination path; None → returns the processed bytes instead
        - key:   32 hex chars, or None → fix-only pass (no decryption)
        - kid:   expected KID (32 hex chars); checked against the file's tenc.
                 Only meaningful together with key — ignored when key is None.
        - meta:  TuneMeta(...) tags written during the same fused pass. Fields
                 left as None (or 0 / empty) keep the file's existing values.
                 cover accepts image bytes/bytearray/memoryview or a path
                 (JPEG/PNG); oversized JPEGs are downscaled to ≤1000px.
        - strict: when True (and key given), a fragment that decrypts to
                 statistical noise (fake/rotated key) raises error 8.

        Returns None when out is a path, bytes when out is None.
        All parameters are keyword-only.
        """
        try:
            return self._lib.proc(input=input, out=out, key=key, kid=kid, meta=meta, strict=strict)
        except Exception as e:
            raise _to_err(e) from None

    async def aproc(
        self,
        *,
        input: bytes | bytearray | memoryview | str,
        out: str | None = None,
        key: str | None = None,
        kid: str | None = None,
        meta: "TuneMeta | None" = None,
        strict: bool = False
    ) -> bytes | None:
        """Async proc — same contract, keyword-only parameters.

        Native coroutine on the extension's own tokio runtime: CPU work runs
        without the GIL, file writes are true async I/O (tokio::fs), and
        asyncio.gather over many calls scales across cores. Cover preparation
        (JPEG decode/resize) also runs on the worker — a big cover never
        blocks the event loop.
        """
        try:
            return await self._lib.aproc(input=input, out=out, key=key, kid=kid, meta=meta, strict=strict)
        except Exception as e:
            raise _to_err(e) from None


def TuneMeta(
    title: str | None = None,
    date: str | None = None,
    artists: list[str] | None = None,
    isrc: str | None = None,
    track: int | None = None,
    track_total: int | None = None,
    disc: int | None = None,
    disc_total: int | None = None,
    album: str | None = None,
    genre: str | None = None,
    lyrics: str | None = None,
    composer: str | None = None,
    cover: bytes | bytearray | memoryview | str | None = None
):
    """Metadata payload for proc/aproc — thin wrapper over the native TuneMeta class.

    0, empty strings and empty artist lists are treated like None.
    The ilst is rebuilt from scratch from these fields (no merge).
    """
    return _native.TuneMeta(title, date, artists, isrc, track, track_total, disc, disc_total,
                           album, genre, lyrics, composer, cover)
