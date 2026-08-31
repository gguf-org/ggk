# The quantizer

The editor's quantizer is `libgguf_quantizer`, a shared library built from
`vendor/engine/quantizer/` and installed at `src/ggk/editor/lib/`. Python
loads it with `ctypes`; the GUI and `ggk editor quantize` are two front ends
onto the same call.

## Why it is a separate library

Encoding is a different job from decoding. The inference runtimes only ever
*read* quantized blocks; the quantizer *writes* them, which needs search,
error minimisation and per-type heuristics that inference has no use for.

What the two must agree on is the on-disk block layout. That agreement is
enforced structurally: `quantizer/src/kernels/qz_*` is the single
implementation of every GGUF block format, and it is compiled into **two**
artifacts —

- `libgguf_quantizer`, the encoder the editor drives, and
- `gk` itself, which links the same sources for the decode side.

Compiling them in rather than duplicating them is what keeps the engine and
the quantizer from ever disagreeing about a block.

## Supported types

Every value accepted by `--type` and by the right-hand side of a
`--tensor-type-rules` entry:

| Class | Types |
| ----- | ----- |
| Float | `f32` · `f16` · `bf16` · `f64` |
| Legacy blocks | `q4_0` · `q4_1` · `q5_0` · `q5_1` · `q8_0` · `q8_1` |
| K-quants | `q2_k` · `q3_k` · `q4_k` · `q5_k` · `q6_k` · `q8_k` |
| I-quants | `iq1_s` · `iq1_m` · `iq2_xxs` · `iq2_xs` · `iq2_s` · `iq3_xxs` · `iq3_s` · `iq4_nl` · `iq4_xs` |
| Ternary | `tq1_0` · `tq2_0` |
| FP4-style | `mxfp4` · `nvfp4` |
| Small blocks | `q1_0` · `q2_0` |
| Integer | `i8` · `i16` · `i32` · `i64` |

## Block sizes matter

Each type quantizes a fixed number of consecutive values into one block. A
tensor can only be converted to a type whose block size divides its **first**
dimension.

| Block size | Types |
| ---------- | ----- |
| 1 (per-element) | `f32`, `f16`, `bf16`, `f64`, `i8`, `i16`, `i32`, `i64` |
| 32 | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `iq4_nl`, `mxfp4` |
| 64 | `nvfp4`, `q2_0` |
| 128 | `q1_0` |
| 256 | all K-quants and I-quants, `tq1_0`, `tq2_0` |

A row of 4096 works with everything. A row of 1152 works with block sizes 32
and 64 but not 256. The GUI greys out impossible targets and reports how many
tensors a batch type skipped; the CLI logs a warning and leaves the tensor
alone.

## From the command line

```bash
ggk editor quantize -m IN -o OUT [--type T] [--tensor-type-rules R]
                                 [--device D] [-t N]
```

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `-m, --model` | required | Input `.gguf` or `.safetensors` |
| `-o, --output` | required | Output `.gguf` |
| `--type TYPE` | — | Default type for every eligible tensor |
| `--tensor-type-rules RULES` | — | `<regex>=<type>` overrides, comma-separated |
| `--device DEVICE` | `auto` | `cpu`, `auto`, or a name from `ggk editor devices` |
| `-t, --threads N` | `0` | Worker threads; 0 auto-detects |

At least one of `--type` / `--tensor-type-rules` is required.

Exit codes: `0` success, `1` error, `2` cancelled.

### Rules

Rules are `<regex>=<type>` pairs, evaluated against the tensor name. The
first match wins; anything unmatched falls back to `--type`, and if there is
no `--type` it keeps its original type.

```bash
# everything q4_k, but keep the output head and token embeddings at f16
ggk editor quantize -m in.gguf -o out.gguf --type q4_k \
    --tensor-type-rules '^output\.=f16,^token_embd\.=f16'

# attention at q6_k, feed-forward at q4_k, the rest untouched
ggk editor quantize -m in.gguf -o out.gguf \
    --tensor-type-rules '\.attn_.*=q6_k,\.ffn_.*=q4_k'

# a diffusion model: VAE at f16, everything else q8_0
ggk editor quantize -m flux.safetensors -o flux-q8.gguf --type q8_0 \
    --tensor-type-rules '^vae\.=f16'
```

The GUI's per-tensor Precision column compiles to exactly this: one anchored
rule per changed tensor, `^<escaped name>$=<type>`.

## Devices

```bash
ggk editor devices
```

```
available devices:
  cpu              CPU (built-in quantization kernels)
  cuda0            NVIDIA GeForce RTX 4050 Laptop GPU
```

Index 0 is always `cpu`. CUDA/HIP devices are named `cuda<N>` / `hip<N>` with
the card's own name as the description; a Metal device is named `metal`.

GPU devices appear only if the quantizer's own device kernels were built:

| Build switch | Adds |
| ------------ | ---- |
| `GGK_CUDA=1` | CUDA (sets `QUANTIZER_CUDA`) |
| `GGK_HIP=1` | HIP (sets `QUANTIZER_HIP`) |
| `-DQUANTIZER_METAL=ON` | Metal — opt-in, not implied by `GGK_METAL` |

These are the quantizer's own kernels, not gk's. A CPU-only quantizer in a
CUDA-enabled build is normal and correct.

`--device` accepts, in order of specificity:

| Value | Resolves to |
| ----- | ----------- |
| `cpu` (or empty) | The CPU kernels |
| `auto` | The first non-CPU device, or `cpu` if there is none |
| An exact name | `cuda0`, `hip1`, `metal` |
| An alias | `cuda`, `hip`, `hipblas`, `rocm`, `gpu` → the first matching accelerator |

An unrecognised value fails with `device '…' is not available in this build`
and logs the list. The GUI defaults to `cpu` for its jobs; the CLI defaults
to `auto`.

## The C API

`vendor/engine/quantizer/src/quantize.h`:

```c
enum gqz_log_level { GQZ_LOG_INFO = 0, GQZ_LOG_WARN = 1, GQZ_LOG_ERROR = 2 };

typedef void (*gqz_log_callback)(enum gqz_log_level level,
                                 const char * message, void * user_data);

struct gqz_params {
    const char * input_path;
    const char * output_path;
    const char * default_type;         /* e.g. "q4_k", or NULL     */
    const char * tensor_type_rules;    /* "<regex>=<type>,…", NULL */
    int          n_threads;            /* 0 = auto                 */
    const char * device;               /* "cpu", "auto", "cuda0"   */
    gqz_log_callback log_cb;           /* NULL = log to stderr     */
    void *       log_user_data;
    int       (* cancel_cb)(void *);   /* return non-zero to stop  */
    void *       cancel_user_data;
};

struct gqz_params gqz_default_params(void);

int          gqz_get_device_count(void);
const char * gqz_get_device_name(int index);
const char * gqz_get_device_description(int index);

int gqz_quantize(const struct gqz_params * params);   /* 0 ok, 2 cancelled */
```

Cancellation is cooperative: `cancel_cb` is polled between tensors, and the
library removes its own partial output before returning `2`.

## From Python

```python
from ggk.editor import quantizer

if not quantizer.is_available():
    raise SystemExit(quantizer.load_error())

def on_log(level, message):
    pct = quantizer.parse_progress(message)      # "[ 42%] …" → 42
    print(f"{pct if pct is not None else '--':>3} {message}")

quantizer.quantize(
    "in.gguf", "out-q4_k.gguf",
    default_type="q4_k",
    tensor_type_rules=r"^output\.=f16",
    n_threads=0,
    device="cpu",
    log_callback=on_log,
    cancel_check=lambda: False,
)
```

Raises `QuantizeError` on failure and `QuantizeCancelled` when
`cancel_check` returns true. `quantizer.list_devices()` returns
`[{"name": …, "description": …}]`.

## Choosing a type

| Type | Bits/weight (approx.) | Use for |
| ---- | --------------------- | ------- |
| `f16` / `bf16` | 16 | Reference; what you quantize *from* |
| `q8_0` | 8.5 | Near-lossless; small models, or quality-critical tensors |
| `q6_k` | 6.6 | Very close to `q8_0` at 3/4 the size |
| `q5_k` | 5.5 | Quality-sensitive work |
| `q4_k` | 4.5 | The usual sweet spot |
| `iq4_xs` | 4.25 | Slightly smaller than `q4_k`; more compute at inference |
| `q3_k` | 3.4 | Models that barely fit |
| `iq2_*` / `iq1_*` | 1–2.5 | Very large models where nothing else fits |
| `mxfp4` / `nvfp4` | ~4 | Hardware with native FP4 paths |
| `tq1_0` / `tq2_0` | 1–2 | Ternary-trained models (BitNet and relatives) |

### Tensors worth keeping high

Two tensors carry disproportionate quality per byte:

- `token_embd.*` — the input embedding table
- `output.*` — the output projection / LM head

Holding those at `f16` or `q6_k` while everything else goes to `q4_k` costs
a few percent of file size and recovers a noticeable amount of quality. That
is what the rules example above does, and roughly what the community "K_M"
mixes encode.

## Limits

- **No importance matrix.** I-quants benefit substantially from an imatrix
  computed over a calibration corpus; this quantizer does not compute or
  consume one. Downloaded IQ quants from projects that do will be better than
  what you produce locally at the same bit width.
- **No requantization gain.** Quantizing an already-quantized model
  dequantizes first and loses more. Always start from `f16`/`bf16`/`f32`.
- **Conversion is out of scope.** There is no PyTorch/HF → GGUF converter
  here. Bring a GGUF or safetensors file.
