# FAQ

### What is `ggk`?

One Python package with three panels — an OpenAI-compatible LLM server, a
diffusion generator, and a GGUF editor with a quantizer — over one C/C++
engine compiled in a single build.

### What is `gk`?

The compute library the whole engine runs on: an independent tensor library
with CPU, CUDA, HIP, Metal and Vulkan backends. It replaced ggml outright.
See [engine/gk.md](engine/gk.md).

### Is there ggml in here?

No. There is no vendored `llama.cpp` checkout and no ggml. What exists is
`gk/compat/`, which implements the historical ggml *C API* on gk — the
structs are layout-identical by construction, the op enums carry the same
values, and every entry point forwards. That is what let a 100k-line runtime
change engines without an edit.

### Why is `pip install` slow?

It compiles the engine from source. There are no pre-built binary wheels,
because the kernels are built for the backends you ask for. CPU takes
minutes; CUDA can take hours.

### Why is my GPU not being used?

GPU backends are opt-in **at install time**:

```bash
GGK_CUDA=1 pip install ggk
```

Check the device banner each engine prints on stderr. See
[hardware.md](hardware.md).

### Can I use one panel without the others?

Yes:

```bash
ggk server      # :8642
ggk diffuser    # :8643
ggk editor      # :8644
```

They share a build but are independent at run time.

### Can I use the engines without the GUI?

Yes — everything after `--` goes to the engine verbatim:

```bash
ggk server engine   -- --model model.gguf --port 8888
ggk diffuser engine -- -m sd.gguf -p "a fox" -o out.png
ggk editor quantize -m in.gguf -o out.gguf --type q4_k
```

### Are my models uploaded anywhere?

No. Everything runs on your machine, and models are addressed by filesystem
path. The one exception is drag & drop into the editor: a browser cannot
reveal a dropped file's path, so those bytes are copied into a temp directory
that is deleted on exit. Use the file picker to avoid even that.

### Is it safe to expose the GUI on my network?

No. The panel APIs are unauthenticated and can list directories, spawn
processes with arguments of your choosing, and write files. Keep them on
`127.0.0.1`.

Exposing the *LLM engine* (`--host 0.0.0.0` in the Server tab, with
`--api-key`) is a much smaller and more reasonable step.

### Where are my settings stored?

In the browser's `localStorage`, per origin. Nothing is written server-side.
See [gui.md § Local state](gui.md#local-state).

### Which models are supported?

- LLMs: [server/models.md](server/models.md) — over 130 architectures
- Diffusion: [diffuser/models.md](diffuser/models.md) — SD 1.x through Flux 2,
  Qwen-Image, Wan, and more

### Can it convert PyTorch / Hugging Face models to GGUF?

No. Conversion is out of scope. Bring a GGUF or safetensors file; the
quantizer converts safetensors → GGUF while quantizing, but there is no
PyTorch pickle loader anywhere in the tree.

### Which quantization should I use?

`q4_k` is the usual sweet spot; `q5_k` or `q6_k` when quality matters more
than size; `q8_0` when size does not matter. Keep `token_embd.*` and
`output.*` higher than the rest — that is most of what the community "K_M"
mixes do. See [editor/quantizer.md § Choosing a type](editor/quantizer.md#choosing-a-type).

### Why are my IQ quants worse than downloaded ones?

I-quants benefit substantially from an importance matrix computed over a
calibration corpus. This quantizer does not compute or consume one, so
downloaded IQ quants from projects that do will be better at the same bit
width. K-quants are much less sensitive to this.

### Can I run several models at once?

The server panel manages one engine at a time. For several models on one
port, use the engine's router mode through the *Edit manually* box:

```bash
ggk server engine -- --models-dir ~/models --models-autoload
```

Or run several panels on different ports.

### Does the LLM server have a chat UI?

Not in this build. The engine can embed one, but the tree ships no asset
bundle, so `GET /` returns 404 — deliberately, since `ggk` serves its own GUI.
Point any OpenAI-compatible client at `http://127.0.0.1:8888/v1`, or supply
your own bundle with `--path` / `GGUF_SERVER_UI_DIR`.

### Is generation reproducible?

Same machine, same build, same seed: yes, bit-for-bit, at any thread count —
gk fixes the accumulation order per row. Across backends or across SIMD
baselines, close but not guaranteed identical. See
[hardware.md § Reproducibility](hardware.md#reproducibility).

### Why two device lists?

Because they genuinely differ. CUDA numbers devices fastest-first, so the
engine's `cuda0` may be `nvidia-smi`'s GPU 1. Always take device names from
the engine. See
[hardware.md § Two different device lists](hardware.md#two-different-device-lists).

### Why is the quantizer a separate library from the runtime?

Encoding is a different job from decoding. What they must agree on is the
on-disk block layout, and that is enforced structurally: the `qz_*` codec is
compiled into both the quantizer and gk, so they cannot disagree. See
[engine/quantizer.md](engine/quantizer.md).

### Why is the default LLM port 8888 and not 8080?

A deliberate difference from upstream `llama-server`, along with the binary
being named `gguf-server`. `--port` and `LLAMA_ARG_PORT` still override it,
and every other flag and environment variable keeps its upstream name.

### Which backends are not supported?

gk implements CPU, CUDA, HIP, Metal and Vulkan. SYCL, CANN, OpenCL, WebGPU,
RPC, zDNN, ZenDNN, OpenVINO, Hexagon, MUSA, virtGPU and ET have no gk
counterpart and are simply absent. Training/autograd is stubbed — the
`*_BACK` ops hold enum slots for compatibility and abort if reached.

### Can I contribute or hack on it?

Yes — [development.md](development.md) covers the layout and the edit loops.

### What license?

MIT, inherited from llama.cpp and stable-diffusion.cpp. `gk/` carries its own
(also MIT) license; third-party components under `thirdparty/` keep theirs.
