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

pub fn parse(data: &[u8], start: usize, end: usize) -> Vec<Box> {
    let mut out: Vec<Box> = Vec::new();
    let mut pos: usize = start;
    while pos + 8 <= end {
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

#[inline]
pub fn write_header(out: &mut Vec<u8>, typ: &[u8; 4], content_len: usize) {
    out.extend_from_slice(&((content_len + 8) as u32).to_be_bytes());
    out.extend_from_slice(typ);
}
