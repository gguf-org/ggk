# The GGUF format, as this editor sees it

Reference for what the editor parses and writes. The implementation is
`src/ggk/editor/gguf.py` — pure Python, stdlib only, no dependency on the
C engine.

## File layout

```
┌─────────────────────────────────────────┐
│ magic          "GGUF"  (uint32 LE)      │
│ version        uint32                   │  1, 2 or 3
│ tensor_count   uint64                   │
│ metadata_count uint64                   │
├─────────────────────────────────────────┤
│ metadata KV pairs × metadata_count      │
│   key      : string                     │
│   type     : uint32  (value type)       │
│   value    : per type                   │
├─────────────────────────────────────────┤
│ tensor infos × tensor_count             │
│   name     : string                     │
│   n_dims   : uint32                     │
│   shape    : uint64 × n_dims            │
│   dtype    : uint32  (tensor type)      │
│   offset   : uint64  (within data)      │
├─────────────────────────────────────────┤
│ padding to `general.alignment`          │
├─────────────────────────────────────────┤
│ tensor data                             │
│   each tensor padded up to alignment    │
└─────────────────────────────────────────┘
```

Everything is little-endian. A string is a `uint64` byte length followed by
that many bytes — **not** null-terminated, and not required to be UTF-8.

`general.alignment` defaults to **32** when absent or unparseable. Tensor
data starts at the first multiple of the alignment at or after the end of the
header, and every tensor's data is padded up to it.

## Metadata value types

| Id | Name | Encoding |
| -- | ---- | -------- |
| 0 | `UINT8` | 1 byte |
| 1 | `INT8` | 1 byte |
| 2 | `UINT16` | 2 bytes |
| 3 | `INT16` | 2 bytes |
| 4 | `UINT32` | 4 bytes |
| 5 | `INT32` | 4 bytes |
| 6 | `FLOAT32` | 4 bytes |
| 7 | `BOOL` | 1 byte, 0 or 1 |
| 8 | `STRING` | uint64 length + bytes |
| 9 | `ARRAY` | uint32 element type + uint64 count + elements |
| 10 | `UINT64` | 8 bytes |
| 11 | `INT64` | 8 bytes |
| 12 | `FLOAT64` | 8 bytes |

Arrays are homogeneous and can nest a `STRING` element type — which is how
tokenizer vocabularies are stored, often with hundreds of thousands of
entries.

### String handling

String values are kept as **raw bytes** through parse, edit and write. Real
tokenizer vocabularies contain byte sequences that are not valid UTF-8;
decoding and re-encoding them would corrupt the model. Only values you
actually edit are re-encoded.

## Tensor types

| Id | Name | Block size | Bytes/block |
| -- | ---- | ---------- | ----------- |
| 0 | `F32` | 1 | 4 |
| 1 | `F16` | 1 | 2 |
| 2 | `Q4_0` | 32 | 18 |
| 3 | `Q4_1` | 32 | 20 |
| 6 | `Q5_0` | 32 | 22 |
| 7 | `Q5_1` | 32 | 24 |
| 8 | `Q8_0` | 32 | 34 |
| 9 | `Q8_1` | 32 | 36 |
| 10 | `Q2_K` | 256 | 84 |
| 11 | `Q3_K` | 256 | 110 |
| 12 | `Q4_K` | 256 | 144 |
| 13 | `Q5_K` | 256 | 176 |
| 14 | `Q6_K` | 256 | 210 |
| 15 | `Q8_K` | 256 | 292 |
| 16 | `IQ2_XXS` | 256 | 66 |
| 17 | `IQ2_XS` | 256 | 74 |
| 18 | `IQ3_XXS` | 256 | 98 |
| 19 | `IQ1_S` | 256 | 50 |
| 20 | `IQ4_NL` | 32 | 18 |
| 21 | `IQ3_S` | 256 | 110 |
| 22 | `IQ2_S` | 256 | 82 |
| 23 | `IQ4_XS` | 256 | 136 |
| 24 | `I8` | 1 | 1 |
| 25 | `I16` | 1 | 2 |
| 26 | `I32` | 1 | 4 |
| 27 | `I64` | 1 | 8 |
| 28 | `F64` | 1 | 8 |
| 29 | `IQ1_M` | 256 | 56 |
| 30 | `BF16` | 1 | 2 |
| 34 | `TQ1_0` | 256 | 54 |
| 35 | `TQ2_0` | 256 | 66 |
| 39 | `MXFP4` | 32 | 17 |
| 40 | `NVFP4` | 64 | 36 |
| 41 | `Q1_0` | 128 | 18 |
| 42 | `Q2_0` | 64 | 18 |

Ids 4, 5 and 31–33 and 36–38 are gaps: types that were removed upstream, or
reserved. A file using one is reported as `Unknown(<id>)` and copied through
byte-for-byte rather than being rejected.

### Tensor byte size

```
rows      = product(shape[1:])          (1 for a 1-D tensor)
row_bytes = (shape[0] / block_size) * bytes_per_block
total     = row_bytes * rows
```

`shape[0]` **must** be divisible by the type's block size. This is the rule
behind every "cannot convert this tensor" message in the editor: a row of
1152 cannot be `Q4_K` (block 256), though it can be `Q8_0` (block 32).

## Parsing

`parse_file()` reads only the header:

1. Read the first 4 MiB.
2. Try to parse. On `EOFError`, double the read size and retry, up to 1 GiB.
3. Compute each tensor's exact byte size from its type and shape.

A tensor with an unknown type has no computable size, so it falls back to the
gap to the next tensor's offset. That over-counts by up to `alignment - 1`
bytes of padding, but copies correctly — which is the point: an unknown type
survives a round-trip intact.

Nothing beyond the header is ever read into memory. Opening a 70 GB model is
as cheap as opening a 70 MB one.

## Writing

`write_rebuilt()` takes a freshly built header plus a **write plan** per
tensor:

```python
@dataclass
class WritePlan:
    size: int
    segments: list = []      # [{"path", "offset", "length"}, …]
    zero_fill: int = 0
```

- A **kept** tensor is one segment pointing into the source file.
- A **merged** tensor is several segments, concatenated in order.
- An **imported** tensor is one segment pointing into the other file.
- A **zero** tensor is `zero_fill` bytes.

The writer emits the header, pads to the alignment, then walks the plans,
copying in 8 MiB chunks and padding each tensor up to the alignment. Peak
memory is one chunk, whatever the model's size.

A progress callback is invoked as bytes are written, and can raise to abort —
which is how Cancel works in the GUI.

## The edit spec

The JSON the GUI accumulates and posts to `/api/save`:

```json
{
  "metadata_edits":    {"general.name": "My Model"},
  "deleted_meta_keys": ["general.url"],
  "new_meta_rows":     [{"key": "general.license", "value": "MIT", "type": 8}],
  "tensor_renames":    {"0": "token_embd.weight"},
  "deleted_tensors":   [17, 18],
  "merges":            [{"name": "merged.weight", "dtype": 1,
                         "shape": [4096, 64], "parts": [3, 4]}],
  "added_tensors":     [{"name": "new.weight", "shape": [4096, 8], "dtype": 0,
                         "source": {"kind": "zeros"}}],
  "order":             ["t:0", "a:0", "t:1", "m:0"]
}
```

Row keys in `order` use:

| Prefix | Refers to |
| ------ | --------- |
| `t:<index>` | An original tensor, by its index in the source file |
| `m:<position>` | `merges[<position>]` |
| `a:<position>` | `added_tensors[<position>]` |

Keys not in `order` are appended in their default order, and unknown keys are
dropped — so a stale `order` degrades gracefully instead of losing tensors.

`added_tensors[].source` is either `{"kind": "zeros"}` or
`{"kind": "external", "path": …, "offset": …, "size": …}` for an import.

Offsets in the output header are recomputed from scratch as the plan is
built; the source offsets are only ever used to read.

## Validation

Before anything is written:

- every tensor name must be non-empty, and
- tensor names must be unique.

Both raise before the output file is created.

## Working with it directly

```python
from ggk.editor import gguf

p = gguf.parse_file("model.gguf")
print(p.version, p.alignment, len(p.tensor_infos), p.file_size)

for key, entry in list(p.metadata.items())[:10]:
    print(key, gguf.VALUE_TYPE_NAME[entry["type"]],
          gguf.format_value(entry["type"], entry["value"]))

for t, size in zip(p.tensor_infos[:5], p.tensor_sizes):
    print(f"{t.name:40} {t.shape} {gguf.quantization_name(t.dtype):8} {size:>12,}")
```
