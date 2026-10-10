"""Minimal pure-Python Parquet reader (flat schemas only).

Supports: thrift compact metadata, UNCOMPRESSED / SNAPPY / GZIP codecs,
DATA_PAGE v1 and v2, DICTIONARY_PAGE, PLAIN and RLE_DICTIONARY encodings,
RLE/bit-packed definition levels, physical types BOOLEAN, INT32, INT64,
FLOAT, DOUBLE, BYTE_ARRAY.
"""
import struct, zlib, sys

# ---------------- thrift compact protocol ----------------
class TReader:
    def __init__(self, b, pos=0):
        self.b, self.p = b, pos
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        r = s = 0
        while True:
            x = self.byte(); r |= (x & 0x7F) << s; s += 7
            if not x & 0x80: return r
    def zz(self):
        v = self.varint(); return (v >> 1) ^ -(v & 1)
    def read_struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0: return out
            t = h & 0x0F; d = h >> 4
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.read_val(t)
    def read_val(self, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: return struct.unpack('b', bytes([self.byte()]))[0]
        if t in (4, 5, 6): return self.zz()
        if t == 7:
            v = struct.unpack('<d', self.b[self.p:self.p+8])[0]; self.p += 8; return v
        if t == 8:
            n = self.varint(); v = self.b[self.p:self.p+n]; self.p += n; return v
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0F
            if n == 15: n = self.varint()
            return [self.read_val(et) if et not in (1, 2) else bool(self.byte() == 1) for _ in range(n)]
        if t == 11:
            n = self.varint()
            if n == 0: return {}
            kv = self.byte(); kt, vt = kv >> 4, kv & 0x0F
            return {self.read_val(kt): self.read_val(vt) for _ in range(n)}
        if t == 12: return self.read_struct()
        raise ValueError(f"bad thrift type {t}")

# ---------------- snappy ----------------
def snappy_decompress(src):
    p = 0; n = s = 0
    while True:
        x = src[p]; p += 1; n |= (x & 0x7F) << s; s += 7
        if not x & 0x80: break
    out = bytearray()
    while p < len(src):
        tag = src[p]; p += 1; t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(src[p:p+nb], 'little'); p += nb
            ln += 1; out += src[p:p+ln]; p += ln
        else:
            if t == 1:
                ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[p]; p += 1
            elif t == 2:
                ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p+2], 'little'); p += 2
            else:
                ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p+4], 'little'); p += 4
            st = len(out) - off
            for i in range(ln): out.append(out[st + i])
    assert len(out) == n, (len(out), n)
    return bytes(out)

import ctypes
_zstd = ctypes.CDLL("libzstd.so.1")
_zstd.ZSTD_decompress.restype = ctypes.c_size_t
_zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
_zstd.ZSTD_isError.restype = ctypes.c_uint
_zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]

def zstd_decompress(data, usize):
    dst = ctypes.create_string_buffer(usize)
    src = ctypes.create_string_buffer(bytes(data), len(data))
    r = _zstd.ZSTD_decompress(dst, usize, src, len(data))
    if _zstd.ZSTD_isError(r): raise ValueError("zstd error")
    return dst.raw[:r]

def decompress(codec, data, usize=None):
    if codec == 0: return data
    if codec == 1: return snappy_decompress(data)
    if codec == 2: return zlib.decompress(data, 16 + zlib.MAX_WBITS)
    if codec == 6: return zstd_decompress(data, usize)
    raise ValueError(f"unsupported codec {codec}")

# ---------------- RLE / bit-packed hybrid ----------------
def rle_decode(buf, bit_width, count):
    out, p = [], 0
    bw_bytes = (bit_width + 7) // 8
    while len(out) < count and p < len(buf):
        h = s = 0
        while True:
            x = buf[p]; p += 1; h |= (x & 0x7F) << s; s += 7
            if not x & 0x80: break
        if h & 1:
            groups = h >> 1; nbytes = groups * bit_width
            chunk = int.from_bytes(buf[p:p+nbytes], 'little'); p += nbytes
            mask = (1 << bit_width) - 1
            for i in range(groups * 8):
                out.append((chunk >> (i * bit_width)) & mask if bit_width else 0)
        else:
            run = h >> 1
            v = int.from_bytes(buf[p:p+bw_bytes], 'little') if bw_bytes else 0; p += bw_bytes
            out.extend([v] * run)
    return out[:count]

# ---------------- plain decoding ----------------
def plain(ptype, buf, n, p=0):
    out = []
    if ptype == 0:  # BOOLEAN bit-packed
        for i in range(n): out.append(bool((buf[p + i // 8] >> (i % 8)) & 1))
        return out, p + (n + 7) // 8
    if ptype == 1:
        return list(struct.unpack_from(f'<{n}i', buf, p)), p + 4 * n
    if ptype == 2:
        return list(struct.unpack_from(f'<{n}q', buf, p)), p + 8 * n
    if ptype == 4:
        return list(struct.unpack_from(f'<{n}f', buf, p)), p + 4 * n
    if ptype == 5:
        return list(struct.unpack_from(f'<{n}d', buf, p)), p + 8 * n
    if ptype == 6:
        for _ in range(n):
            ln = struct.unpack_from('<I', buf, p)[0]; p += 4
            out.append(buf[p:p+ln]); p += ln
        return out, p
    raise ValueError(f"unsupported physical type {ptype}")

def read_parquet(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'PAR1' and b[-4:] == b'PAR1'
    mlen = struct.unpack('<I', b[-8:-4])[0]
    meta = TReader(b, len(b) - 8 - mlen).read_struct()
    schema = meta[2]
    leaves = []
    for el in schema[1:]:
        if el.get(5):  # num_children -> nested group
            raise ValueError("nested schema not supported")
        leaves.append({'name': el[4].decode(), 'type': el.get(1), 'opt': el.get(3, 0) == 1,
                       'conv': el.get(6), 'logical': el.get(10)})
    cols = {l['name']: [] for l in leaves}
    for rg in meta[4]:
        for ci, cc in enumerate(rg[1]):
            md = cc[3]
            leaf = leaves[ci]; ptype = md[1]; codec = md[4]
            nvals_total = md[5]
            start = md.get(11, md[9])  # dictionary_page_offset or data_page_offset
            if md.get(11) is not None:
                start = min(md[11], md[9])
            p = start; got = 0; dictionary = None
            while got < nvals_total:
                tr = TReader(b, p); ph = tr.read_struct(); p = tr.p
                ptype_page = ph[1]; usize = ph[2]; csize = ph[3]
                raw = b[p:p+csize]; p += csize
                if ptype_page == 2:  # dictionary page
                    data = decompress(codec, raw, usize)
                    dictionary, _ = plain(ptype, data, ph[7][1])
                    continue
                if ptype_page == 0:  # data page v1
                    dh = ph[5]; n = dh[1]; enc = dh[2]
                    data = decompress(codec, raw, usize); q = 0
                    if leaf['opt']:
                        ln = struct.unpack_from('<I', data, q)[0]; q += 4
                        defs = rle_decode(data[q:q+ln], 1, n); q += ln
                    else:
                        defs = [1] * n
                    body = data[q:]
                elif ptype_page == 3:  # data page v2
                    dh = ph[8]; n = dh[3]; enc = dh[4]
                    dl = dh[5]; rl = dh[6]
                    lev = raw[:rl + dl]; rest = raw[rl + dl:]
                    is_comp = dh.get(7, True)
                    body = decompress(codec, rest, usize - rl - dl) if is_comp else rest
                    defs = rle_decode(lev[rl:], 1, n) if (leaf['opt'] and dl) else [1] * n
                else:
                    raise ValueError(f"page type {ptype_page}")
                nn = sum(1 for d in defs if d)
                if enc in (2, 8):  # dictionary indices
                    bw = body[0]
                    idx = rle_decode(body[1:], bw, nn)
                    vals = [dictionary[i] for i in idx]
                elif enc == 0:
                    vals, _ = plain(ptype, body, nn)
                elif enc == 3 and ptype == 0:  # RLE booleans
                    ln = struct.unpack_from('<I', body, 0)[0]
                    vals = [bool(v) for v in rle_decode(body[4:4+ln], 1, nn)]
                else:
                    raise ValueError(f"encoding {enc} for type {ptype}")
                it = iter(vals)
                for d in defs:
                    cols[leaf['name']].append(next(it) if d else None)
                got += n
    # decode strings
    for l in leaves:
        if l['type'] == 6:
            cols[l['name']] = [v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v for v in cols[l['name']]]
    nrows = meta[3]
    return cols, nrows

if __name__ == '__main__':
    for path in sys.argv[1:]:
        cols, n = read_parquet(path)
        print(path, n, {k: len(v) for k, v in cols.items()})
        print({k: v[:2] for k, v in cols.items()})
