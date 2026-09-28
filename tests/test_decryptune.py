"""End-to-end tests for the decryptune native extension.

Builds a synthetic fragmented MP4 (sidx + encrypted/plain/tfhd-fallback
fragments + moov with enca/sinf/pssh), encrypts payloads with real AES-128-CTR,
and validates the output byte-for-byte.

Run:  python3 -m pytest tests/
      DECRYPTUNE_WHEEL=1 python3 -m pytest tests/    # against the installed wheel
"""

import asyncio
import importlib.util
import os
import random
import struct
import sys

import pytest
from Crypto.Cipher import AES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.environ.get("DECRYPTUNE_WHEEL"):
    # installed wheel: import from site-packages — strip cwd so a same-named
    # standalone wrapper file in the repo cannot shadow the package
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.getcwd()) != os.getcwd()]
    from decryptune import DecrypTune, DecrypTuneError, TuneMeta
else:
    # repo checkout: the standalone wrapper + the extension file by path
    _spec = importlib.util.spec_from_file_location("decryptune_wrapper", os.path.join(_ROOT, "decryptune.py"))
    _wrapper = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_wrapper)
    DecrypTune, DecrypTuneError, TuneMeta = _wrapper.DecrypTune, _wrapper.DecrypTuneError, _wrapper.TuneMeta

KEY_HEX = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
KEY = bytes.fromhex(KEY_HEX)
FAKE_KEY_HEX = KEY_HEX[:-1] + ("0" if KEY_HEX[-1] != "0" else "1")

PNG = b"\x89PNG\r\n\x1a\n" + bytes(32)
FULL_META_KWARGS = dict(title="Test Song", date="2026", artists=["A One", "B Two"],
                        isrc="IRABC1234567", track=3, track_total=12, disc=1, disc_total=2,
                        album="Album X", genre="Pop", lyrics="la la", composer="C One",
                        cover=PNG)


# ---- synthetic file builders ------------------------------------------------

def u32(v): return struct.pack(">I", v)
def i32(v): return struct.pack(">i", v)
def u16(v): return struct.pack(">H", v)
def box(typ, payload): return u32(len(payload) + 8) + typ + payload


def full_box(typ, version, flags, payload):
    return box(typ, bytes([version]) + u32(flags)[1:] + payload)


def make_senc(ivs):
    body = u32(len(ivs)) + b"".join(ivs)  # flags 0: full-sample IVs only
    return full_box(b"senc", 0, 0, body)


def make_trun(sizes, data_offset, with_sizes=True):
    flags = 0x1 | (0x200 if with_sizes else 0)
    body = u32(len(sizes)) + i32(data_offset)
    if with_sizes:
        body += b"".join(u32(s) for s in sizes)
    return full_box(b"trun", 0, flags, body)


def make_tfhd(default_size=None):
    if default_size is None:
        return full_box(b"tfhd", 0, 0, u32(1))                       # track_ID only
    return full_box(b"tfhd", 0, 0x10, u32(1) + u32(default_size))    # + default_sample_size


def make_saiz(count):
    return full_box(b"saiz", 0, 0, u32(0) + u32(0) + bytes([0]) + u32(count) + bytes(count))


def make_saio():
    return full_box(b"saio", 0, 0, u32(1) + u32(0))


def make_moof(seq, traf_children):
    mfhd = full_box(b"mfhd", 0, 0, u32(seq))
    traf = box(b"traf", b"".join(traf_children))
    return box(b"moof", mfhd + traf)


def make_esds():
    asc = b"\x12\x10"
    dsc = bytes([0x05, len(asc)]) + asc
    dec = bytes([0x04, 13 + len(dsc), 0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0]) + dsc
    sl = bytes([0x06, 0x01, 0x02])
    es = bytes([0x03, 3 + len(dec) + len(sl), 0, 0]) + dec + sl
    return full_box(b"esds", 0, 0, es)


def make_sinf():
    frma = box(b"frma", b"mp4a")
    schm = full_box(b"schm", 0, 0, b"cenc" + u32(0x10000))
    # layout mirrors a real tenc v0: ver/flags(4) + reserved(2) +
    # crypt_byte(1) + IV_len(1) + default_KID(16) = 24 content bytes
    tenc_body = bytes([0, 0, 1, 8]) + bytes(16)
    tenc = full_box(b"tenc", 0, 0, tenc_body)
    schi = box(b"schi", tenc)
    return box(b"sinf", frma + schm + schi)


def make_enca():
    audio_common = (
        bytes(6) + u16(1) +          # reserved + data_reference_index
        u16(0) + u16(0) + u32(0) +   # reserved x2
        u16(2) + u16(16) + u16(0) + u16(0) + u32(44100 << 16)  # channels..samplerate
    )
    # sinf lives inside the protected sample entry per spec
    return box(b"enca", audio_common + make_esds() + make_sinf())


def make_stsd():
    body = u32(1) + make_enca()  # entry_count; ver/flags via full_box
    return full_box(b"stsd", 0, 0, body)


def make_moov():
    mvhd = full_box(b"mvhd", 0, 0,
        u32(0) + u32(0) + u32(1000) + u32(12000) + u32(0x10000) +
        u16(0x0100) + u16(0) + bytes(8) + bytes(36) + bytes(24) + u32(2))
    tkhd = full_box(b"tkhd", 0, 7,
        u32(0) + u32(0) + u32(1) + u32(12000) + u32(0) + u32(0) +
        u16(0) + u16(0) + u16(0x0100) + u16(0) + bytes(36) + u32(0) + u32(0))
    mdhd = full_box(b"mdhd", 0, 0, u32(0) + u32(0) + u32(44100) + u32(529200) + u16(0x55C4) + u16(0))
    hdlr = full_box(b"hdlr", 0, 0, u32(0) + b"soun" + bytes(12) + b"\x00")
    smhd = full_box(b"smhd", 0, 0, u16(0) + u16(0))
    dref = full_box(b"dref", 0, 0, u32(1) + full_box(b"url ", 0, 1, b""))
    stbl = box(b"stbl", make_stsd() +
        full_box(b"stts", 0, 0, u32(1) + u32(1024) + u32(44100 // 1024 * 12)) +
        full_box(b"stsc", 0, 0, u32(0)) +
        full_box(b"stsz", 0, 0, u32(0) + u32(0)) +
        full_box(b"stco", 0, 0, u32(0)))
    dinf = box(b"dinf", dref)
    minf = box(b"minf", smhd + dinf + stbl)
    mdia = box(b"mdia", mdhd + hdlr + minf)
    trak = box(b"trak", tkhd + mdia)
    pssh = full_box(b"pssh", 0, 0, bytes(16) + u32(0) + u32(0))
    return box(b"moov", mvhd + trak + pssh)


def make_sidx(ref_sizes):
    entries = b"".join(
        u32(sz) + u32(4000) + u32(0x90000000)  # SAP word with top bit set
        for sz in ref_sizes
    )
    body = (u32(1) + u32(1000) + u32(0) + u32(0) + u16(0) +
            u16(len(ref_sizes)) + entries)
    return full_box(b"sidx", 0, 0, body)


def ctr_crypt(key, iv8, data):
    iv16 = iv8 + bytes(8)
    c = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=int.from_bytes(iv16, "big"))
    return c.encrypt(data)


def build_sample_file():
    """ftyp + sidx + [enc frag] + [plain frag] + [tfhd-fallback frag] + moov."""
    ftyp = box(b"ftyp", b"M4A " + u32(0) + b"M4A " + b"mp42" + b"isom")

    n = 4
    sizes = [20000] * n  # payloads large enough for the chi2 judge threshold
    plain = [bytes([0x11 * i]) * 20000 for i in range(1, n + 1)]
    ivs = [bytes([j, 0xA, 3, 4, 5, 6, 7, 8]) for j in range(n)]
    enc = [ctr_crypt(KEY, ivs[i], plain[i]) for i in range(n)]

    # trun data_offset: first byte of sample data relative to moof start
    def set_offset(moof, off):
        i = moof.find(b"trun")           # position of the type string
        return moof[:i + 12] + i32(off) + moof[i + 16:]

    moof1 = make_moof(1, [make_tfhd(), make_trun(sizes, 0), make_senc(ivs),
                          make_saiz(n), make_saio()])
    moof2 = make_moof(2, [make_tfhd(), make_trun(sizes, 0)])
    moof3 = make_moof(3, [make_tfhd(sizes[0]), make_trun(sizes, 0, with_sizes=False),
                          make_senc(ivs)])

    # trun data_offset: first byte of sample data relative to moof start
    def set_offset(moof, off):
        i = moof.find(b"trun")           # position of the type string
        return moof[:i + 12] + i32(off) + moof[i + 16:]

    moof1 = set_offset(moof1, len(moof1) + 8)
    moof2 = set_offset(moof2, len(moof2) + 8)
    moof3 = set_offset(moof3, len(moof3) + 8)

    mdat1 = box(b"mdat", b"".join(enc))
    mdat2 = box(b"mdat", b"".join(plain))          # never encrypted
    mdat3 = box(b"mdat", b"".join(enc))            # tfhd-fallback fragment

    sidx = make_sidx([len(moof1), len(moof2), len(moof3)])
    return (ftyp + sidx + moof1 + mdat1 + moof2 + mdat2 + moof3 + mdat3 + make_moov(),
            {1: plain, 2: plain, 3: plain})


# ---- box-tree helpers -------------------------------------------------------

def walk(data):
    out, pos = [], 0
    while pos + 8 <= len(data):
        (size,) = struct.unpack_from(">I", data, pos)
        typ = data[pos + 4:pos + 8]
        size = len(data) - pos if size == 0 else size
        out.append((typ, pos, size))
        pos += size
    return out


def find(data, typ):
    return [b for b in walk(data) if b[0] == typ]


def find_boxes(d, start, end, typ):
    res, pos = [], start
    while pos + 8 <= end:
        (size,) = struct.unpack_from(">I", d, pos)
        t = d[pos + 4:pos + 8]
        if size == 0: size = end - pos
        if size < 8 or pos + size > end: break
        if t == typ: res.append((pos, size))
        pos += size
    return res


def ilst_items(d):
    moov = find_boxes(d, 0, len(d), b"moov")[0]
    udta = find_boxes(d, moov[0] + 8, moov[0] + moov[1], b"udta")[0]
    meta = find_boxes(d, udta[0] + 8, udta[0] + udta[1], b"meta")[0]
    ilst = find_boxes(d, meta[0] + 12, meta[0] + meta[1], b"ilst")[0]
    items = {}
    pos, end = ilst[0] + 8, ilst[0] + ilst[1]
    while pos + 8 <= end:
        (size,) = struct.unpack_from(">I", d, pos)
        fourcc = d[pos + 4:pos + 8]
        items.setdefault(fourcc, []).append(d[pos + 8:pos + size])
        pos += size
    return items


def data_payload(item_content):
    # item content begins with its 'data' child box
    (size,) = struct.unpack_from(">I", item_content, 0)
    assert item_content[4:8] == b"data"
    return item_content[16:size]


def fb_child(d, typ):
    pos, end = 0, len(d)
    while pos + 8 <= end:
        (cs,) = struct.unpack_from(">I", d, pos)
        if d[pos + 4:pos + 8] == typ:
            return d[pos + 12:pos + cs]  # ver/flags(4) + payload
        pos += cs
    return None


def jpeg_dims(j):
    i = 2
    while i + 9 < len(j):
        if j[i] != 0xFF:
            i += 1; continue
        marker = j[i + 1]
        if marker in (0xC0, 0xC2):
            h, w = struct.unpack_from(">HH", j, i + 5)
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
        else:
            (seg,) = struct.unpack_from(">H", j, i + 2)
            i += 2 + seg
    return None


def assert_decrypted(out, expected_plain):
    """Full structural validation of a decrypted + sanitized sample."""
    top = [t for t, _, _ in walk(out)]
    assert top[:2] == [b"ftyp", b"sidx"]
    assert b"moof" in top and b"mdat" in top and b"moov" in top

    moofs = find(out, b"moof")
    mdats = find(out, b"mdat")
    assert len(moofs) == 3 and len(mdats) == 3

    for idx, (moof, mdat) in enumerate(zip(moofs, mdats), 1):
        _, s, size = moof
        content = out[s + 8:s + size]
        assert b"senc" not in content and b"saiz" not in content and b"saio" not in content
        assert b"trun" in content
        payload = out[mdat[1] + 8:mdat[1] + mdat[2]]
        assert payload == b"".join(expected_plain[idx])

    # trun data_offset of frag1 points at its mdat content
    moof1 = moofs[0]
    trun_pos = out.find(b"trun", moof1[1], moof1[1] + moof1[2])
    (doff,) = struct.unpack_from(">i", out, trun_pos + 12)
    assert doff == moof1[2] + 8

    # sidx referenced sizes patched, SAP top bit preserved
    sidx = find(out, b"sidx")[0]
    sc = out[sidx[1] + 8:sidx[1] + sidx[2]]
    for k, moof in enumerate(moofs):
        (word,) = struct.unpack_from(">I", sc, 24 + k * 12)
        assert word == moof[2]
    (sap,) = struct.unpack_from(">I", sc, 24 + 8)
    assert sap & 0x80000000 == 0x80000000

    moov = find(out, b"moov")[0]
    mc = out[moov[1]:moov[1] + moov[2]]
    assert b"pssh" not in mc
    assert b"enca" not in mc and b"sinf" not in mc and b"frma" not in mc
    assert b"mp4a" in mc and b"esds" in mc


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture(scope="session")
def sp():
    return DecrypTune() if os.environ.get("DECRYPTUNE_WHEEL") else DecrypTune(os.path.join(_ROOT, "libdecryptune.so"))


@pytest.fixture(scope="session")
def sample():
    return build_sample_file()   # (file bytes, expected plaintext per fragment)


@pytest.fixture(scope="session")
def src_file(sample, tmp_path_factory):
    path = tmp_path_factory.mktemp("src") / "in.m4a"
    path.write_bytes(sample[0])
    return str(path)


@pytest.fixture(scope="session")
def full_meta(sp):
    # must be created after a DecrypTune instance — TuneMeta binds to the loaded extension
    return TuneMeta(**FULL_META_KWARGS)


@pytest.fixture(scope="session")
def tagged(sp, sample, full_meta):
    data, _ = sample
    return sp.proc(input=data, key=KEY_HEX, meta=full_meta)


# ---- error codes ------------------------------------------------------------

def test_error_invalid_key(sp, src_file):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, key="nothex")
    assert e.value.code == 1


def test_error_invalid_kid(sp, src_file):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, key=KEY_HEX, kid="nothex")
    assert e.value.code == 2


def test_error_missing_input(sp, tmp_path):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=str(tmp_path / "missing.m4a"), key=KEY_HEX)
    assert e.value.code == 3


def test_error_not_an_mp4(sp):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=__file__, key=KEY_HEX)
    assert e.value.code == 4


def test_error_kid_mismatch(sp, src_file):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, key=KEY_HEX, kid="f" * 32)
    assert e.value.code == 7


def test_error_write_failed(sp, src_file, tmp_path):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, out=str(tmp_path / "no_such_dir" / "x.m4a"), key=KEY_HEX)
    assert e.value.code == 6


def test_error_invalid_out_type(sp, src_file):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, out=123, key=KEY_HEX)
    assert e.value.code == 10


def test_error_bad_cover(sp, src_file):
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, key=KEY_HEX, meta=TuneMeta(cover=b"notanimage"))
    assert e.value.code == 12


def test_error_cover_missing(sp, src_file):
    # additive code 13 — a missing cover path names the cover, not the input
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, key=KEY_HEX, meta=TuneMeta(cover="no_such_cover.jpg"))
    assert e.value.code == 13
    assert "cover file not found" in e.value.message


def test_path_out_pathlib(sp, src_file, tmp_path):
    # out accepts os.PathLike exactly like input
    dst = tmp_path / "out_pathlib.m4a"
    assert sp.proc(input=src_file, out=dst, key=KEY_HEX) is None
    assert dst.read_bytes() == sp.proc(input=src_file, key=KEY_HEX)


class _BytesPath:
    """os.fspath objects may legally return bytes — must behave like a str path."""

    def __init__(self, p):
        self.p = str(p)

    def __fspath__(self):
        return os.fsencode(self.p)


def test_fspath_bytes_input_and_out(sp, src_file, tmp_path):
    dst = tmp_path / "out_fspath.m4a"
    assert sp.proc(input=_BytesPath(src_file), out=_BytesPath(dst), key=KEY_HEX) is None
    assert dst.read_bytes() == sp.proc(input=src_file, key=KEY_HEX)


def test_out_is_directory_no_tmp_litter(sp, src_file, tmp_path):
    # rename onto an existing directory fails — the tmp file must not be left behind
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=src_file, out=str(out_dir), key=KEY_HEX)
    assert e.value.code == 6
    assert list(tmp_path.glob("*.tmp")) == []


def test_aproc_out_is_directory_no_tmp_litter(sp, src_file, tmp_path):
    out_dir = tmp_path / "aoutdir"
    out_dir.mkdir()
    with pytest.raises(DecrypTuneError) as e:
        asyncio.run(sp.aproc(input=src_file, out=str(out_dir), key=KEY_HEX))
    assert e.value.code == 6
    assert list(tmp_path.glob("*.tmp")) == []


def test_error_tags_need_fragmented(sp):
    # a non-empty stco dangles once moov grows with tags -> refused
    stco = full_box(b"stco", 0, 0, u32(1) + u32(100))
    stco_file = box(b"moov", box(b"trak", box(b"mdia", box(b"minf", box(b"stbl", stco)))))
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=stco_file, meta=TuneMeta(title="X"))
    assert e.value.code == 11


# ---- proc -------------------------------------------------------------------

def test_path_out(sp, sample, src_file, tmp_path):
    data, expected = sample
    dst = tmp_path / "out.m4a"
    assert sp.proc(input=src_file, out=str(dst), key=KEY_HEX, kid="00" * 16) is None
    assert_decrypted(dst.read_bytes(), expected)


def test_bytes_out(sp, sample):
    data, expected = sample
    out = sp.proc(input=data, key=KEY_HEX)
    assert isinstance(out, bytes)
    assert_decrypted(out, expected)


def test_buffer_inputs(sp, sample):
    data, _ = sample
    ref = sp.proc(input=data, key=KEY_HEX)
    assert sp.proc(input=bytearray(data), key=KEY_HEX) == ref
    assert sp.proc(input=memoryview(data), key=KEY_HEX) == ref
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=123, key=KEY_HEX)
    assert e.value.code == 9


def test_fix_only(sp, sample):
    data, _ = sample
    fix = sp.proc(input=data)
    mdat_in = find(data, b"mdat")[0]
    mdat_fix = find(fix, b"mdat")[0]
    assert fix[mdat_fix[1] + 8:mdat_fix[1] + mdat_fix[2]] == data[mdat_in[1] + 8:mdat_in[1] + mdat_in[2]]
    assert b"senc" not in fix and b"pssh" not in fix


def test_kid_ignored_without_key(sp, sample):
    data, _ = sample
    fix = sp.proc(input=data)
    assert sp.proc(input=data, kid="f" * 32) == fix   # wrong kid + no key must NOT raise


def test_strict(sp, sample):
    data, _ = sample
    good = sp.proc(input=data, key=KEY_HEX)
    assert sp.proc(input=data, key=KEY_HEX, strict=True) == good
    with pytest.raises(DecrypTuneError) as e:
        sp.proc(input=data, key=FAKE_KEY_HEX, strict=True)
    assert e.value.code == 8
    assert "fragments:" in str(e.value) and "chi2=" in str(e.value)


def test_fuzz_no_native_panics(sp, sample):
    data, _ = sample
    random.seed(42)
    for _ in range(300):
        b = bytearray(data)
        for _ in range(random.randint(1, 8)):
            b[random.randrange(len(b))] = random.randrange(256)
        try:
            sp.proc(input=bytes(b), key=KEY_HEX)
        except DecrypTuneError:
            pass


@pytest.mark.parametrize(("version", "content_len"), [(0, 23), (1, 28), (1, 31)])
def test_hostile_truncated_sidx(sp, version, content_len):
    # a v1 header is longer than v0 — the count word must not be read past the box
    content = bytes([version, 0, 0, 0]) + bytes(content_len - 4)
    hostile = box(b"ftyp", b"M4A isom") + box(b"sidx", content)
    assert isinstance(sp.proc(input=hostile), bytes)


# ---- metadata ---------------------------------------------------------------

def test_udta_first(sp, tagged):
    # Windows' MP4 property handler reads the FIRST udta/meta subtree
    moov = find(tagged, b"moov")[0]
    assert tagged[moov[1] + 12:moov[1] + 16] == b"udta"


def test_metadata_items(tagged):
    items = ilst_items(tagged)
    assert data_payload(items[b"\xa9nam"][0]) == b"Test Song"
    assert data_payload(items[b"\xa9day"][0]) == b"2026"
    assert [data_payload(i) for i in items[b"\xa9ART"]] == [b"A One", b"B Two"]
    assert data_payload(items[b"aART"][0]) == b"A One"
    # freeform: mean/name are standalone tag boxes
    ff = items[b"----"][0]
    assert fb_child(ff, b"mean") == b"com.apple.iTunes"
    assert fb_child(ff, b"name") == b"ISRC"
    assert fb_child(ff, b"data")[4:] == b"IRABC1234567"   # skip 4-byte locale
    assert data_payload(items[b"trkn"][0])[2:6] == bytes([0, 3, 0, 12])
    assert data_payload(items[b"disk"][0])[2:6] == bytes([0, 1, 0, 2])
    assert data_payload(items[b"\xA9alb"][0]) == b"Album X"
    assert data_payload(items[b"\xA9gen"][0]) == b"Pop"
    assert data_payload(items[b"\xA9lyr"][0]) == b"la la"
    assert data_payload(items[b"\xA9wrt"][0]) == b"C One"
    cov = items[b"covr"][0]
    (csize,) = struct.unpack_from(">I", cov, 0)
    (ctype,) = struct.unpack_from(">I", cov, 8)
    assert ctype == 14 and cov[16:csize] == PNG


def test_metadata_wipe(sp, tagged):
    items = ilst_items(sp.proc(input=tagged, key=KEY_HEX, meta=TuneMeta(title="Renamed")))
    assert data_payload(items[b"\xa9nam"][0]) == b"Renamed"
    assert b"\xa9day" not in items and b"\xa9ART" not in items and b"covr" not in items


def test_metadata_two_udta(sp, sample):
    # a second udta must not survive the wipe (it would shadow the fresh subtree)
    data, _ = sample
    extra = box(b"udta", box(b"\xA9day", struct.pack(">II", 1, 0) + b"1999"))
    moov = find(data, b"moov")[0]
    duped = data[:moov[1] + 8] + extra + data[moov[1] + 8:]
    items = ilst_items(sp.proc(input=duped, key=KEY_HEX, meta=TuneMeta(title="Solo")))
    assert data_payload(items[b"\xa9nam"][0]) == b"Solo"
    assert b"\xa9day" not in items


# ---- cover ------------------------------------------------------------------

FIXTURES = os.path.join(_ROOT, "tests", "fixtures")


def test_cover_resize(sp, sample):
    data, _ = sample
    tagged = sp.proc(input=data, key=KEY_HEX, meta=TuneMeta(cover=os.path.join(FIXTURES, "big.jpg")))
    cov = ilst_items(tagged)[b"covr"][0]
    (csize,) = struct.unpack_from(">I", cov, 0)
    (ctype,) = struct.unpack_from(">I", cov, 8)
    w, h = jpeg_dims(cov[16:csize])
    assert ctype == 13 and max(w, h) <= 1000


def test_cover_passthrough(sp, sample):
    data, _ = sample
    with open(os.path.join(FIXTURES, "tiny.jpg"), "rb") as f:
        tiny = f.read()
    tagged = sp.proc(input=data, key=KEY_HEX, meta=TuneMeta(cover=os.path.join(FIXTURES, "tiny.jpg")))
    cov = ilst_items(tagged)[b"covr"][0]
    assert cov[16:struct.unpack_from(">I", cov, 0)[0]] == tiny


def test_cover_bytearray(sp, sample):
    data, _ = sample
    with open(os.path.join(FIXTURES, "tiny.jpg"), "rb") as f:
        tiny = f.read()
    tagged = sp.proc(input=data, key=KEY_HEX, meta=TuneMeta(cover=bytearray(tiny)))
    cov = ilst_items(tagged)[b"covr"][0]
    assert cov[16:struct.unpack_from(">I", cov, 0)[0]] == tiny


# ---- async ------------------------------------------------------------------

def test_async_meta_matches_sync(sp, sample, full_meta, tagged):
    data, _ = sample
    assert asyncio.run(sp.aproc(input=data, key=KEY_HEX, meta=full_meta)) == tagged


def test_async_big_cover_on_worker(sp, sample):
    # cover prepare must run on the worker and stay byte-identical to sync
    data, _ = sample
    big = os.path.join(FIXTURES, "big.jpg")
    sync = sp.proc(input=data, key=KEY_HEX, meta=TuneMeta(cover=big))
    assert asyncio.run(sp.aproc(input=data, key=KEY_HEX, meta=TuneMeta(cover=big))) == sync


def test_async_paths(sp, sample, src_file, tmp_path):
    data, _ = sample
    dst = tmp_path / "async.m4a"
    sync = sp.proc(input=data, key=KEY_HEX)
    assert asyncio.run(sp.aproc(input=data, key=KEY_HEX)) == sync
    assert asyncio.run(sp.aproc(input=src_file, out=str(dst), key=KEY_HEX, kid="00" * 16)) is None
    assert dst.read_bytes() == sync


def test_async_strict_fake_key(sp, sample):
    data, _ = sample
    with pytest.raises(DecrypTuneError) as e:
        asyncio.run(sp.aproc(input=data, key=FAKE_KEY_HEX, strict=True))
    assert e.value.code == 8


# ---- idempotence ------------------------------------------------------------

def test_idempotence(sp, sample):
    data, _ = sample
    out = sp.proc(input=data, key=KEY_HEX)
    assert sp.proc(input=out) == out   # second pass byte-identical
