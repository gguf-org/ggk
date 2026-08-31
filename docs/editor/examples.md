# Editor panel — examples

## Command line

### Quantize a whole model

```bash
ggk editor quantize -m llama-3-8b-f16.gguf -o llama-3-8b-q4_k.gguf --type q4_k
```

### A K_M-style mix

Keep the two tensors that carry the most quality per byte at higher
precision:

```bash
ggk editor quantize \
    -m llama-3-8b-f16.gguf \
    -o llama-3-8b-q4_k_m.gguf \
    --type q4_k \
    --tensor-type-rules '^output\.=q6_k,^token_embd\.=q6_k'
```

### Per-layer-type precision

```bash
ggk editor quantize \
    -m model-f16.gguf -o model-mixed.gguf \
    --tensor-type-rules '\.attn_.*=q6_k,\.ffn_.*=q4_k,^output\.=f16'
```

Tensors matched by no rule keep their original type, because `--type` is
absent.

### A diffusion model

```bash
ggk editor quantize \
    -m flux1-dev.safetensors -o flux1-dev-q8_0.gguf \
    --type q8_0 \
    --tensor-type-rules '^vae\.=f16'
```

Safetensors in, GGUF out.

### Use the GPU

```bash
ggk editor devices
ggk editor quantize -m in.gguf -o out.gguf --type q4_k --device cuda0 -t 0
```

Available only if the quantizer's own CUDA/HIP/Metal kernels were built —
see [quantizer.md § Devices](quantizer.md#devices).

---

## In the GUI

### Retitle a model

1. Open the file.
2. Type `general.name` in the search box.
3. Edit the value.
4. **Save** → accept the `_edited.gguf` path.

Metadata-only edits still rebuild the file, because the header length changes
and every tensor offset shifts with it. Tensor data is copied, not touched.

### Fix a broken chat template

1. Search for `tokenizer.chat_template`.
2. Paste a working Jinja template into the textarea.
3. Save.

Faster alternative if you only need it for one run: leave the file alone and
pass `--chat-template-file` to the server panel instead.

### Rename a whole tensor namespace

1. Open **Find & Replace in Names**.
2. Find `model.diffusion_model.`, Replace with `` (empty).
3. Check the live match count, then **Replace All**.
4. Save.

Tick **Regex** for anchored patterns like `^blk\.(\d+)\.` — the replacement
supports capture-group references.

### Strip tensors you do not need

1. Search for the prefix, e.g. `vision.`
2. Click Delete on each row (or use the checkboxes and delete them together).
3. Save.

The stats bar's *New size* updates as you go, so you can see what you are
saving before committing.

### Add a zero-filled tensor

1. **Zero Tensor…**
2. Name `mymodel.bias`, precision `F32`, shape `4096`.
3. **Add Tensor** → Save.

If the first dimension is not a multiple of the chosen type's block size, the
modal says so and refuses.

### Copy tensors in from another file

1. **Import from GGUF…** → pick the source file.
2. Filter, check the tensors you want, **Add Selected**.
3. Save.

Data is streamed directly from the source file at save time. Names must not
collide with existing ones.

### Merge tensors

1. Check two or more tensors in the table.
2. **Merge Selected**.
3. Give the result a name; it takes the parts' type and concatenates along
   the last axis.
4. Save.

The source tensors are dropped from the output.

### Quantize with per-tensor control

1. Open the **Quantization** drawer.
2. Set **Weight type** to `q4_k` — every eligible tensor gets it.
3. In the table, set individual **Precision** cells for the ones you want
   different: `token_embd.weight` → `f16`, `output.weight` → `q6_k`.
4. **Save** — the title changes to *Save & Quantize as…*.

The plan compiles to one anchored rule per changed tensor, so exactly the set
you picked is converted.

### Reorder tensors

Drag the ☰ handle. Order affects the sequence tensors are written in, which
matters for loaders that stream sequentially, and for diffing two files.

---

## From Python

### Inspect a file

```python
from ggk.editor import gguf

p = gguf.parse_file("model.gguf")

print(f"GGUF v{p.version}, alignment {p.alignment}")
print(f"{len(p.metadata)} metadata keys, {len(p.tensor_infos)} tensors")
print(f"{p.file_size / 1e9:.2f} GB, data starts at {p.tensor_data_offset}")

arch = p.metadata.get("general.architecture")
if arch:
    print("architecture:", gguf.format_value(arch["type"], arch["value"]))
```

### Total bytes per tensor type

```python
from collections import Counter
from ggk.editor import gguf

p = gguf.parse_file("model.gguf")
by_type = Counter()
for t, size in zip(p.tensor_infos, p.tensor_sizes):
    by_type[gguf.quantization_name(t.dtype)] += size

for name, total in by_type.most_common():
    print(f"{name:10} {total / 1e9:8.3f} GB")
```

### Quantize programmatically

```python
from ggk.editor import quantizer

quantizer.quantize(
    "in.gguf", "out.gguf",
    default_type="q4_k",
    tensor_type_rules=r"^output\.=q6_k,^token_embd\.=q6_k",
    device="cpu",
    log_callback=lambda level, msg: print(msg),
)
```

### Batch-quantize a directory

```python
import pathlib
from ggk.editor import quantizer

src = pathlib.Path("~/models/f16").expanduser()
dst = pathlib.Path("~/models/q4_k").expanduser()
dst.mkdir(parents=True, exist_ok=True)

for path in sorted(src.glob("*.gguf")):
    out = dst / path.name.replace("-f16", "-q4_k")
    if out.exists():
        continue
    print("→", out.name)
    quantizer.quantize(
        str(path), str(out),
        default_type="q4_k",
        tensor_type_rules=r"^output\.=q6_k,^token_embd\.=q6_k",
        log_callback=lambda lvl, m: None,
    )
```

### Edit metadata without the GUI

```python
from ggk.editor import gguf
from ggk.editor.edits import build_edited_header, compute_final_tensors

parsed = gguf.parse_file("model.gguf")
edits = {
    "metadata_edits": {"general.name": "Renamed Model"},
    "deleted_meta_keys": ["general.url"],
    "new_meta_rows": [{"key": "general.license", "value": "MIT", "type": gguf.STRING}],
}

finals, plans = compute_final_tensors(parsed, edits)
header = build_edited_header(parsed, edits, finals)
gguf.write_rebuilt("model-renamed.gguf", header, plans, parsed.alignment,
                   lambda done, total: print(f"\r{done * 100 // total}%", end=""))
print()
```

---

## Driving the panel from a script

```bash
ggk editor --no-browser --port 8644 &

# inspect
curl -s http://127.0.0.1:8644/api/open \
  -H 'Content-Type: application/json' \
  -d '{"path": "/models/llama-3.gguf"}' | python3 -m json.tool | head -40

# quantize
JOB=$(curl -s http://127.0.0.1:8644/api/quantize \
  -H 'Content-Type: application/json' \
  -d '{"input_path": "/models/llama-3-f16.gguf",
       "output_path": "/models/llama-3-q4_k.gguf",
       "quant": {"type": "q4_k", "device": "cpu", "threads": 0}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')

while :; do
  curl -s "http://127.0.0.1:8644/api/job/$JOB" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["status"], d["progress"], d["stage"])'
  curl -s "http://127.0.0.1:8644/api/job/$JOB" | grep -q '"status": "running"' || break
  sleep 2
done
```

Full schemas: [api.md](api.md).
