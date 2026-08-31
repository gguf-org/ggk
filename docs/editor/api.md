# Editor panel — local JSON API

The API of the **panel**. It parses GGUF headers, runs save/quantize jobs,
and serves the GUI.

Base URL:

- standalone: `http://127.0.0.1:8644/api/…`
- under the unified GUI: `http://127.0.0.1:8640/editor/api/…`

Responses are JSON with `Cache-Control: no-store`. Errors are
`{"error": "message"}` with a 4xx/5xx status.

---

## `GET /api/status`

```json
{
  "version": "0.5.1",
  "quantizer_available": true,
  "quantizer_error": null,
  "quant_types": ["f32", "f16", "q4_0", "…"],
  "initial_file": "/Users/you/models/llama-3.gguf",
  "home": "/Users/you"
}
```

`initial_file` is the path given on the command line (`ggk editor
model.gguf`), or `null`. `quantizer_available: false` disables every
precision control in the GUI and shows `quantizer_error` as a banner.

---

## `GET /api/devices`

```json
{ "devices": [
  {"name": "cpu",   "description": "CPU (built-in quantization kernels)"},
  {"name": "cuda0", "description": "NVIDIA GeForce RTX 4050 Laptop GPU"}
]}
```

The devices compiled into `libgguf_quantizer`. Index 0 is always `cpu`. See
[quantizer.md § Devices](quantizer.md#devices). Empty when the library did
not load.

---

## `POST /api/open`

Parse a file's header.

```json
{ "path": "/Users/you/models/llama-3.gguf" }
```

```json
{
  "path": "/Users/you/models/llama-3.gguf",
  "file_name": "llama-3.gguf",
  "file_size": 4661211808,
  "version": 3,
  "alignment": 32,
  "tensor_data_offset": 741376,
  "metadata": [
    {"key": "general.architecture", "type": 8, "type_name": "STRING",
     "value": "llama", "array_len": null},
    {"key": "tokenizer.ggml.tokens", "type": 9, "type_name": "[STRING]",
     "value": "[\"<unk>\", \"<s>\", …]", "array_len": 128256}
  ],
  "tensors": [
    {"index": 0, "name": "token_embd.weight", "shape": [4096, 128256],
     "dtype": 12, "dtype_name": "Q4_K", "size": 302514176, "offset": 0}
  ]
}
```

Only the header is read. `value` is a display string — arrays are truncated
at 30 elements — while `array_len` gives the true length.

`404` if the path is not a file.

---

## `POST /api/browse`

```json
{ "path": "/Users/you/models" }
```

```json
{
  "path": "/Users/you/models",
  "parent": "/Users/you",
  "entries": [
    {"name": "quants", "path": "/Users/you/models/quants", "is_dir": true},
    {"name": "llama-3.gguf", "path": "/Users/you/models/llama-3.gguf",
     "is_dir": false, "size": 4661211808}
  ]
}
```

Directories, then `.gguf` / `.safetensors` files. Dotfiles hidden. A file
path lists its parent directory.

---

## `POST /api/upload?name=FILENAME`

The one route that takes a raw body: the bytes of a drag & dropped file. A
browser cannot reveal a dropped file's path, so the bytes are streamed
(1 MiB at a time) into a per-server temp directory that is removed on exit.

```
POST /api/upload?name=model.gguf
Content-Length: 4661211808
<raw bytes>
```

```json
{ "path": "/tmp/gguf-editor-drop-ab12/xyz/model.gguf" }
```

Rejects anything not ending in `.gguf` / `.safetensors`, and an empty body.
A failed upload removes its partial file.

Prefer `/api/browse` + `/api/open` for large models — nothing is copied
that way.

---

## `POST /api/save`

Rebuild the file with edits applied. Returns a job id immediately.

```json
{
  "input_path": "/Users/you/models/llama-3.gguf",
  "output_path": "/Users/you/models/llama-3_edited.gguf",
  "edits": {
    "metadata_edits":    {"general.name": "My Model"},
    "deleted_meta_keys": [],
    "new_meta_rows":     [],
    "tensor_renames":    {},
    "deleted_tensors":   [],
    "merges":            [],
    "added_tensors":     [],
    "order":             null
  }
}
```

```json
{ "job_id": "a1b2c3d4e5f6" }
```

The edit spec is documented in
[gguf-format.md § The edit spec](gguf-format.md#the-edit-spec).

---

## `POST /api/quantize`

Quantize without applying edits.

```json
{
  "input_path": "/Users/you/models/llama-3.gguf",
  "output_path": "/Users/you/models/llama-3-q4_k.gguf",
  "quant": {
    "type": "q4_k",
    "rules": "^output\\.=f16,^token_embd\\.=f16",
    "device": "cpu",
    "threads": 0
  }
}
```

```json
{ "job_id": "a1b2c3d4e5f6" }
```

At least one of `type` / `rules` must be present.

---

## `POST /api/save-quantize`

Both, in order: apply the edits into a temp file next to the output, then
quantize that into the output, then remove the temp file. When the edit spec
is empty the first step is skipped and the source is used directly.

```json
{
  "input_path": "/Users/you/models/llama-3.gguf",
  "output_path": "/Users/you/models/llama-3-q4_k.gguf",
  "edits": { "…": "…" },
  "quant": { "rules": "^blk\\.0\\.attn_q\\.weight$=q6_k", "device": "cpu", "threads": 0 }
}
```

```json
{ "job_id": "a1b2c3d4e5f6" }
```

This is what the GUI's **Save** button posts when the precision plan is
non-empty.

---

## `GET /api/job/{id}`

Poll a job. `?after=N` returns only log lines from index `N` onward.

```json
{
  "id": "a1b2c3d4e5f6",
  "kind": "save+quantize",
  "status": "running",
  "stage": "Step 2/2: quantizing…",
  "progress": 42,
  "log": ["[ 42%] blk.12.ffn_down.weight → q4_k"],
  "log_next": 87,
  "error": null,
  "output_path": null
}
```

`kind` is `save`, `quantize` or `save+quantize`. `status` is `running`,
`completed`, `error` or `cancelled`. `output_path` is filled in on success.

Progress comes from the writer's byte counter for a save, and from the
quantizer's `[ 42%]` log prefix for a quantize.

`404` for an unknown job id.

---

## `POST /api/job/{id}/cancel`

```json
{ "ok": true }
```

Sets the job's cancel flag. A save raises out of its progress callback; a
quantize returns through the library's cancel callback and removes its own
partial output. Either way the job ends as `cancelled` with no leftover
output file.

---

## Notes for API consumers

- **Jobs are in-memory and per-process.** Restarting the panel forgets them.
- **Paths are server-side.** Except `/api/upload`, nothing crosses the wire.
- **No authentication**, and `/api/save*` writes to any path the process can
  write to. Keep it on `127.0.0.1`.
- **Save is not in-place.** Always write to a new path; the editor will
  happily overwrite the file it is reading, and that ends badly.
