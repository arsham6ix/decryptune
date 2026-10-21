#[derive(Clone, Copy)]
pub struct Box {
    pub typ           : [u8; 4],
    pub start         : usize,
    pub content_start : usize,
    pub end           : usize
}

impl Box {
    #[inline]
    pub fn is(&self, t: &[u8; 4]) -> bool { &self.typ == t }
}

// hostile box spam: each parsed box costs 32 bytes of vec, so an absurd
// structure would multiply the input size many-fold. Parsing stops at the cap;
// callers check saturation via len() == MAX_BOXES and treat the input as
// invalid or pass it through rather than copying a structure no real file has.
// 2M top-level boxes would already be a 64MB vec for a 16MB input.
pub const MAX_BOXES: usize = 2_000_000;

pub fn parse(data: &[u8], start: usize, end: usize) -> Vec<Box> {
    let mut out: Vec<Box> = Vec::new();
    let mut pos: usize = start;
    while pos + 8 <= end {
        if out.len() >= MAX_BOXES { break; }
        let (size, hlen): (usize, usize) = match u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) {
            0                    => (end - pos, 8),
            1 if pos + 16 <= end => (u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize, 16),
            1                    => break,
            n                    => (n as usize, 8)
        };
        if size < hlen || pos + size > end { break; }
        let mut typ: [u8; 4] = [0u8; 4];
        typ.copy_from_slice(&data[pos + 4..pos + 8]);
        out.push(Box { typ, start: pos, content_start: pos + hlen, end: pos + size });
        pos += size;
    }
    out
}

// header length a box with this content size needs — content beyond the u32
// range switches to the 64-bit largesize form instead of wrapping the size
#[inline]
pub fn header_len(content_len: usize) -> usize {
    if content_len + 8 > u32::MAX as usize { 16 } else { 8 }
}

#[inline]
pub fn write_header(out: &mut Vec<u8>, typ: &[u8; 4], content_len: usize) {
    if content_len + 8 > u32::MAX as usize {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(typ);
        out.extend_from_slice(&((content_len as u64) + 16).to_be_bytes());
    } else {
        out.extend_from_slice(&((content_len + 8) as u32).to_be_bytes());
        out.extend_from_slice(typ);
    }
}

// output sink — Buf writes into an exactly-sized buffer, Count only measures.
// The measure pass runs the identical walk with the same sizing decisions, so
// the buffer can never over- or under-fill.
pub enum Writer<'a> {
    Buf { buf: &'a mut [u8], pos: usize },
    Count(usize)
}

impl Writer<'_> {
    pub fn len(&self) -> usize {
        match self { Writer::Buf { pos, .. } => *pos, Writer::Count(n) => *n }
    }

    pub fn put(&mut self, bytes: &[u8]) {
        match self {
            Writer::Buf { buf, pos } => {
                buf[*pos..*pos + bytes.len()].copy_from_slice(bytes);
                *pos += bytes.len();
            }
            Writer::Count(n) => *n += bytes.len()
        }
    }

    // 32-bit box header (size + fourcc) — or the 64-bit largesize form when the
    // content cannot be expressed in 32 bits (rebuilt >4GB containers)
    pub fn header(&mut self, typ: &[u8; 4], content_len: usize) {
        match self {
            Writer::Buf { buf, pos } => {
                if content_len + 8 > u32::MAX as usize {
                    buf[*pos..*pos + 4].copy_from_slice(&1u32.to_be_bytes());
                    buf[*pos + 4..*pos + 8].copy_from_slice(typ);
                    buf[*pos + 8..*pos + 16].copy_from_slice(&((content_len as u64) + 16).to_be_bytes());
                    *pos += 16;
                } else {
                    buf[*pos..*pos + 4].copy_from_slice(&((content_len + 8) as u32).to_be_bytes());
                    buf[*pos + 4..*pos + 8].copy_from_slice(typ);
                    *pos += 8;
                }
            }
            Writer::Count(n) => *n += header_len(content_len)
        }
    }
}
