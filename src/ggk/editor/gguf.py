"""GGUF binary format parser and writer (pure Python, stdlib only).

Port of the reference implementation in gguf-chrome-extension/gguf-parser.js:
header-only parsing (tensor data stays on disk), metadata/tensor edits, and a
streaming rebuild that copies tensor data straight between files.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Set

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# ── value types ──────────────────────────────────────────────────────────────

UINT8, INT8, UINT16, INT16 = 0, 1, 2, 3
UINT32, INT32, FLOAT32, BOOL = 4, 5, 6, 7
STRING, ARRAY, UINT64, INT64, FLOAT64 = 8, 9, 10, 11, 12

VALUE_TYPE_NAME = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16",
    4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
    8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64", 12: "FLOAT64",
}

QUANTIZATION_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0",
    7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
    11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

# (block_size, type_size) per ggml_type id — mirrors ggml_blck_size /
# ggml_type_size; needed for exact tensor byte sizes when rebuilding.
GGML_TYPE_TRAITS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
    7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110),
    12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
    39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18),
}

# dtypes a user can pick for a new zero-filled tensor (order = UI order)
ZERO_TENSOR_DTYPES = [0, 1, 30, 8, 2, 3, 6, 7, 15, 14, 13, 12, 11, 10,
                      19, 29, 16, 17, 22, 18, 21, 23, 20, 39, 40, 41, 42]


def quantization_name(dtype: int) -> str:
    return QUANTIZATION_NAME.get(dtype, f"Unknown({dtype})")


def type_traits(dtype: int):
    return GGML_TYPE_TRAITS.get(dtype)


def tensor_byte_size(dtype: int, shape: List[int]) -> int:
    """Exact byte size of a tensor's data (ggml_row_size() * nrows)."""
    traits = GGML_TYPE_TRAITS.get(dtype)
    if traits is None:
        raise ValueError(f"Unknown tensor type id {dtype}")
    block_size, type_size = traits
    ne0 = shape[0] if shape else 1
    if ne0 % block_size != 0:
        raise ValueError(
            f"Row size {ne0} is not divisible by block size {block_size} "
            f"of {quantization_name(dtype)}"
        )
    rows = 1
    for d in shape[1:]:
        rows *= d
    return (ne0 // block_size) * type_size * rows


# ── parsed model ─────────────────────────────────────────────────────────────

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: int
    offset: int  # offset within the tensor data section


@dataclass
class ParsedGGUF:
    version: int
    # key -> {"type": int, "value": ...}; STRING values are raw bytes so
    # non-UTF-8 tokenizer entries survive round-trips; ARRAY values are
    # {"elem_type": int, "items": [...]}
    metadata: Dict[str, Dict[str, Any]]
    tensor_infos: List[TensorInfo]
    tensor_data_offset: int
    alignment: int
    path: str = ""
    file_size: int = 0
    tensor_sizes: List[int] = field(default_factory=list)


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def _unpack(self, fmt: str, size: int):
        if self.off + size > len(self.buf):
            raise EOFError("truncated header")
        v = struct.unpack_from(fmt, self.buf, self.off)[0]
        self.off += size
        return v

    def u8(self):  return self._unpack("<B", 1)
    def i8(self):  return self._unpack("<b", 1)
    def u16(self): return self._unpack("<H", 2)
    def i16(self): return self._unpack("<h", 2)
    def u32(self): return self._unpack("<I", 4)
    def i32(self): return self._unpack("<i", 4)
    def u64(self): return self._unpack("<Q", 8)
    def i64(self): return self._unpack("<q", 8)
    def f32(self): return self._unpack("<f", 4)
    def f64(self): return self._unpack("<d", 8)

    def raw_string(self) -> bytes:
        n = self.u64()
        if self.off + n > len(self.buf):
            raise EOFError("truncated header")
        v = self.buf[self.off:self.off + n]
        self.off += n
        return v

    def string(self) -> str:
        return self.raw_string().decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype == UINT8:   return self.u8()
        if vtype == INT8:    return self.i8()
        if vtype == UINT16:  return self.u16()
        if vtype == INT16:   return self.i16()
        if vtype == UINT32:  return self.u32()
        if vtype == INT32:   return self.i32()
        if vtype == FLOAT32: return self.f32()
        if vtype == BOOL:    return self.u8() != 0
        if vtype == STRING:  return self.raw_string()
        if vtype == UINT64:  return self.u64()
        if vtype == INT64:   return self.i64()
        if vtype == FLOAT64: return self.f64()
        if vtype == ARRAY:
            elem_type = self.u32()
            count = self.u64()
            items = [self.value(elem_type) for _ in range(count)]
            return {"elem_type": elem_type, "items": items}
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def _parse_buffer(buf: bytes) -> ParsedGGUF:
    r = _Reader(buf)
    magic = r.u32()
    if magic != GGUF_MAGIC:
        raise ValueError("Invalid GGUF file (bad magic bytes)")
    version = r.u32()
    if version < 1 or version > 3:
        raise ValueError(f"Unsupported GGUF version: {version}")

    tensor_count = r.u64()
    metadata_count = r.u64()

    metadata: Dict[str, Dict[str, Any]] = {}
    for _ in range(metadata_count):
        key = r.string()
        vtype = r.u32()
        metadata[key] = {"type": vtype, "value": r.value(vtype)}

    tensor_infos: List[TensorInfo] = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        shape = [r.u64() for _ in range(n_dims)]
        dtype = r.u32()
        offset = r.u64()
        tensor_infos.append(TensorInfo(name, shape, dtype, offset))

    align_entry = metadata.get("general.alignment")
    alignment = 32
    if align_entry is not None:
        try:
            a = int(align_entry["value"])
            if a > 0:
                alignment = a
        except (TypeError, ValueError):
            pass

    tensor_data_offset = -(-r.off // alignment) * alignment
    return ParsedGGUF(version, metadata, tensor_infos, tensor_data_offset, alignment)


HEADER_READ_INITIAL = 4 * 1024 * 1024
HEADER_READ_MAX = 1024 * 1024 * 1024


def parse_file(path: str) -> ParsedGGUF:
    """Parse the GGUF header of ``path`` without loading tensor data."""
    file_size = os.path.getsize(path)
    size = HEADER_READ_INITIAL
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(min(size, file_size))
            try:
                parsed = _parse_buffer(buf)
                break
            except EOFError:
                if len(buf) >= file_size:
                    raise ValueError("File appears truncated or malformed")
                if size >= HEADER_READ_MAX:
                    raise ValueError(
                        f"GGUF header exceeds {HEADER_READ_MAX // 1024 // 1024} MB")
                size = min(size * 2, HEADER_READ_MAX)
    parsed.path = os.path.abspath(path)
    parsed.file_size = file_size
    parsed.tensor_sizes = compute_tensor_sizes(parsed, file_size)
    return parsed


def compute_tensor_sizes(parsed: ParsedGGUF, total_file_size: int) -> List[int]:
    """Exact data size of every tensor; unknown dtypes fall back to the gap to
    the next tensor's offset (includes alignment padding, but copies correctly)."""
    data_len = total_file_size - parsed.tensor_data_offset
    order = sorted(range(len(parsed.tensor_infos)),
                   key=lambda i: parsed.tensor_infos[i].offset)
    sizes = [0] * len(parsed.tensor_infos)
    for k, i in enumerate(order):
        t = parsed.tensor_infos[i]
        end = parsed.tensor_infos[order[k + 1]].offset if k + 1 < len(order) else data_len
        try:
            sizes[i] = tensor_byte_size(t.dtype, t.shape)
        except ValueError:
            sizes[i] = max(0, end - t.offset)
    return sizes


# ── writer ───────────────────────────────────────────────────────────────────

class _Writer:
    def __init__(self):
        self.chunks: List[bytes] = []

    def _pack(self, fmt: str, v):
        self.chunks.append(struct.pack(fmt, v))

    def u8(self, v):  self._pack("<B", v)
    def i8(self, v):  self._pack("<b", v)
    def u16(self, v): self._pack("<H", v)
    def i16(self, v): self._pack("<h", v)
    def u32(self, v): self._pack("<I", v)
    def i32(self, v): self._pack("<i", v)
    def u64(self, v): self._pack("<Q", v)
    def i64(self, v): self._pack("<q", v)
    def f32(self, v): self._pack("<f", v)
    def f64(self, v): self._pack("<d", v)

    def string(self, s):
        b = s if isinstance(s, (bytes, bytearray)) else str(s).encode("utf-8")
        self.u64(len(b))
        self.chunks.append(bytes(b))

    def value(self, vtype: int, v):
        if vtype == UINT8:   self.u8(int(v))
        elif vtype == INT8:  self.i8(int(v))
        elif vtype == UINT16: self.u16(int(v))
        elif vtype == INT16: self.i16(int(v))
        elif vtype == UINT32: self.u32(int(v))
        elif vtype == INT32: self.i32(int(v))
        elif vtype == FLOAT32: self.f32(float(v))
        elif vtype == BOOL:  self.u8(1 if v else 0)
        elif vtype == STRING: self.string(v)
        elif vtype == UINT64: self.u64(int(v))
        elif vtype == INT64: self.i64(int(v))
        elif vtype == FLOAT64: self.f64(float(v))
        elif vtype == ARRAY:
            self.u32(v["elem_type"])
            self.u64(len(v["items"]))
            for item in v["items"]:
                self.value(v["elem_type"], item)
        else:
            raise ValueError(f"Unknown type: {vtype}")

    def build(self) -> bytes:
        return b"".join(self.chunks)


def build_header(version: int, metadata: Dict[str, Dict[str, Any]],
                 tensors: List[TensorInfo], alignment: int) -> bytes:
    """Serialize a complete GGUF header (padded to the alignment boundary).
    Tensor offsets must already be relative to the data section start."""
    w = _Writer()
    w.u32(GGUF_MAGIC)
    w.u32(version)
    w.u64(len(tensors))
    w.u64(len(metadata))
    for key, entry in metadata.items():
        w.string(key)
        w.u32(entry["type"])
        w.value(entry["type"], entry["value"])
    for t in tensors:
        w.string(t.name)
        w.u32(len(t.shape))
        for dim in t.shape:
            w.u64(dim)
        w.u32(t.dtype)
        w.u64(t.offset)
    header = w.build()
    padded_len = -(-len(header) // alignment) * alignment
    return header + b"\x00" * (padded_len - len(header))


# ── edited-value parsing (mirrors parseEditedValue in gguf-parser.js) ────────

def _parse_array_edited_value(elem_type: int, edited: str):
    s = edited.strip()
    s = re.sub(r"^\[", "", s)
    s = re.sub(r"\]$", "", s)
    items = []
    if s:
        for p in (part.strip() for part in s.split(",")):
            if not p:
                continue
            if elem_type in (FLOAT32, FLOAT64):
                items.append(float(p))
            elif elem_type == BOOL:
                items.append(p.lower() == "true")
            elif elem_type == STRING:
                items.append(p.encode("utf-8"))
            else:
                items.append(int(p, 0))
    return {"elem_type": elem_type, "items": items}


def parse_edited_value(vtype: int, original_value, edited: str):
    """Parse a user-edited string back into a typed value; falls back to the
    original value if parsing fails. New-row array types are 100 + elem_type."""
    try:
        if 100 <= vtype < 120:
            return _parse_array_edited_value(vtype - 100, edited)
        if vtype == STRING:
            return edited.encode("utf-8")
        if vtype == BOOL:
            return edited.strip().lower() == "true"
        if vtype in (FLOAT32, FLOAT64):
            return float(edited)
        if vtype == ARRAY:
            return _parse_array_edited_value(original_value["elem_type"], edited)
        return int(edited.strip(), 0)
    except (ValueError, TypeError, KeyError):
        return original_value


# ── display formatting (mirrors formatValue) ─────────────────────────────────

def format_value(vtype: int, value, max_array_elements: int = 30) -> str:
    if vtype == ARRAY:
        items = value["items"]
        shown = [format_value(value["elem_type"], v) for v in items[:max_array_elements]]
        more = f", … (+{len(items) - max_array_elements})" if len(items) > max_array_elements else ""
        return f"[{', '.join(shown)}{more}]"
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.7g}"
    return str(value)


# ── edit application + streaming rebuild ─────────────────────────────────────

def build_updated_metadata(metadata: Dict[str, Dict[str, Any]],
                           edited: Dict[str, str],
                           deleted_keys: Set[str],
                           new_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    updated: Dict[str, Dict[str, Any]] = {}
    for key, entry in metadata.items():
        if key in deleted_keys:
            continue
        if key in edited:
            updated[key] = {
                "type": entry["type"],
                "value": parse_edited_value(entry["type"], entry["value"], edited[key]),
            }
        else:
            updated[key] = entry
    for row in new_rows:
        k = str(row.get("key", "")).strip()
        if k and k not in updated:
            rtype = int(row["type"])
            actual = ARRAY if rtype >= 100 else rtype
            updated[k] = {"type": actual,
                          "value": parse_edited_value(rtype, None, str(row.get("value", "")))}
    return updated


@dataclass
class WritePlan:
    """How to produce one tensor's data: file segments then zero fill."""
    size: int
    segments: List[Dict[str, Any]] = field(default_factory=list)  # {path, offset, length}
    zero_fill: int = 0


COPY_CHUNK = 8 * 1024 * 1024


def _copy_range(dst: BinaryIO, src_path: str, offset: int, length: int,
                progress: Optional[Callable[[int], None]] = None,
                open_files: Optional[Dict[str, BinaryIO]] = None):
    if open_files is not None and src_path in open_files:
        src = open_files[src_path]
    else:
        src = open(src_path, "rb")
        if open_files is not None:
            open_files[src_path] = src
    src.seek(offset)
    remaining = length
    while remaining > 0:
        chunk = src.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise IOError(f"Unexpected EOF reading {src_path}")
        dst.write(chunk)
        remaining -= len(chunk)
        if progress:
            progress(len(chunk))
    if open_files is None:
        src.close()


def _write_zeros(dst: BinaryIO, count: int,
                 progress: Optional[Callable[[int], None]] = None):
    zeros = b"\x00" * min(count, COPY_CHUNK)
    remaining = count
    while remaining > 0:
        take = min(remaining, len(zeros))
        dst.write(zeros[:take])
        remaining -= take
        if progress:
            progress(take)


def write_rebuilt(output_path: str, header: bytes, plans: List[WritePlan],
                  alignment: int,
                  progress: Optional[Callable[[int, int], None]] = None):
    """Write header + tensor data (streamed from source files) + padding.

    ``progress(written, total)`` is invoked as bytes are written.
    """
    total = len(header) + sum(-(-p.size // alignment) * alignment for p in plans)
    written = [0]

    def bump(n: int):
        written[0] += n
        if progress:
            progress(written[0], total)

    open_files: Dict[str, BinaryIO] = {}
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "wb") as out:
            out.write(header)
            bump(len(header))
            for plan in plans:
                for seg in plan.segments:
                    _copy_range(out, seg["path"], seg["offset"], seg["length"],
                                bump, open_files)
                if plan.zero_fill > 0:
                    _write_zeros(out, plan.zero_fill, bump)
                pad = -(-plan.size // alignment) * alignment - plan.size
                if pad > 0:
                    _write_zeros(out, pad, bump)
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    finally:
        for f in open_files.values():
            f.close()
