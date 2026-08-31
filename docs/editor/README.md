# Editor panel

A GGUF metadata and tensor editor with a built-in quantizer.

```bash
ggk                     # Editor tab of the unified GUI
ggk editor              # standalone, http://127.0.0.1:8644/
ggk editor model.gguf   # open a file straight away
```

## In this section

| Document | What it covers |
| -------- | -------------- |
| [panels.md](panels.md) | The editing surface: tables, drawers, modals |
| [quantizer.md](quantizer.md) | Quantization types, rules, devices, the C API |
| [gguf-format.md](gguf-format.md) | GGUF layout, value types, tensor type traits |
| [api.md](api.md) | The panel's own local JSON API |
| [examples.md](examples.md) | Worked examples |

## What it can do

| | |
| --- | --- |
| **Inspect** | Every metadata key/value and every tensor's name, shape, type, size and offset |
| **Edit metadata** | Change values in place, add rows, delete rows |
| **Edit tensors** | Rename (individually or by find & replace), delete, reorder |
| **Merge tensors** | Concatenate selected tensors along the last axis into one |
| **Add tensors** | Zero-filled tensors, or tensors imported from another GGUF file |
| **Quantize** | A batch weight type and/or per-tensor precisions, in one pass |
| **Undo/redo** | Full history over every edit above |

Opens `.gguf` and `.safetensors`. Writes GGUF.

## How it works

```
browser  ──HTTP──▶  ggk editor panel (Python, stdlib only)
                        ├── gguf.py       header parse / rebuild, pure Python
                        └── quantizer.py  ctypes → libgguf_quantizer
                                              │
                                              ▼
                                          qz_* block codec (C/C++)
```

**Only the header is parsed.** Metadata and tensor descriptors are read from
the front of the file; tensor *data* is never loaded into memory. Editing a
70 GB model is as fast as editing a 70 MB one, right up until you save.

**Saving streams file-to-file.** The new header is built, then each tensor's
bytes are copied from the source file to the destination in 8 MiB chunks, so
peak memory stays flat regardless of model size.

**Quantizing is a separate library.** `libgguf_quantizer` is loaded by
ctypes; it is not the inference runtime. Encoding is a different job from
decoding, so it is a different library — but the two share the same `qz_*`
block codec sources, compiled into both, so they can never disagree about a
GGUF block layout.

## Files

| Path | What |
| ---- | ---- |
| `src/ggk/editor/__main__.py` | The `ggk editor` CLI |
| `src/ggk/editor/server.py` | HTTP backend: static files, `/api/*`, job runner |
| `src/ggk/editor/gguf.py` | GGUF parser and streaming writer (pure Python) |
| `src/ggk/editor/edits.py` | Applies a GUI edit spec to a parsed file |
| `src/ggk/editor/quantizer.py` | ctypes bindings for the quantizer library |
| `src/ggk/editor/static/` | The web GUI |
| `src/ggk/editor/lib/libgguf_quantizer.*` | The compiled library (installed by the build) |

## Save flows

There are three, chosen automatically by what you changed:

| You changed | What runs |
| ----------- | --------- |
| Metadata / structure only | **Save** — the file is rebuilt with your edits |
| Precision only | **Quantize** — the quantizer reads the original and writes the output |
| Both | **Save & Quantize** — edits are applied to a temp file first, then quantized into the output; the temp file is removed either way |

Every one of them runs as a background job with a progress percentage and a
log, and can be cancelled. A cancelled job leaves no partial output — the
quantizer removes its own, and a cancelled save raises before the file is
finished.

## Drag & drop

Dropped files are the one exception to "nothing is uploaded". A browser
cannot reveal a dropped file's path, so `POST /api/upload` streams the bytes
into a temp directory, which is removed when the panel exits. Outputs for a
dropped file default to your home directory rather than that temp directory,
which would vanish.

For large models, use the file picker instead — it hands over a path and
copies nothing.
