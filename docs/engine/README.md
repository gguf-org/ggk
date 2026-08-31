# The ggk engine

One C/C++ source tree at `vendor/engine/`, one CMake build, three artifacts:

| Artifact | Panel that uses it | What it is |
| -------- | ------------------ | ---------- |
| `gguf-server` | Server | The OpenAI-compatible LLM HTTP server |
| `diffusion` | Diffuser | The image/video/audio generation CLI |
| `libgguf_quantizer` | Editor | The quantizer, loaded by ctypes |

There is **no vendored `llama.cpp` checkout and no `ggml` anywhere**. The
graphs the runtimes build are evaluated by **gk**, an independent tensor
library, and everything — kernels, GGUF runtime, diffusion engine, HTTP layer
— is compiled from sources in this one directory.

## Architecture

```
┌─────────────┬──────────────┬────────────────┐
│ app/        │ diffusion/   │ quantizer/     │   the three front ends
│ HTTP server │ generation   │ encoder lib    │
├─────────────┼──────────────┤                │
│ src/        │              │                │   GGUF LLM runtime
│ common/     │              │                │   args, chat, sampling
│ mtmd/       │              │                │   multimodal projectors
├─────────────┴──────────────┤                │
│ gk/compat/                 │                │   the historical ggml C API
├────────────────────────────┼────────────────┤
│ gk/                        │  qz_* codec    │   kernels + block formats
└────────────────────────────┴────────────────┘
```

Read it as two rules:

**Nothing above `gk/compat/` knows gk exists.** `src/`, `mtmd/`, `common/`
and `diffusion/` include the same `ggml.h` / `ggml-backend.h` / `gguf.h`
headers they always did and call the same functions. The compat layer is a
thin translation: the structs are layout-identical by construction (with a
static assert that says so), the op enums carry the same values, and every
entry point forwards. That is what let a 100k-line runtime change engines
without an edit.

**The format layer is shared, not duplicated.** gk does not define the GGUF
block layouts. They live in `quantizer/src/kernels/qz_*` and are compiled
into both the quantizer (which writes them) and gk (which reads them). A
tensor the quantizer writes and a tensor the engine reads can therefore never
disagree about a block.

## Directory map

| Directory | What it holds |
| --------- | ------------- |
| `gk/` | The compute kernels: CPU plus optional CUDA / HIP / Metal / Vulkan |
| `gk/compat/` | The historical ggml C API, implemented on gk |
| `src/` | The GGUF model runtime (`libllama`): loader, graphs, KV cache, sampling |
| `include/` | The runtime's public headers (`llama.h`, `llama-cpp.h`) |
| `common/` | Argument parsing, chat templates (Jinja), sampling presets, logging |
| `mtmd/` | Multimodal projectors — vision and audio inputs |
| `app/` | The HTTP server: routes, task queue, slots, streaming, router mode |
| `ui/` | Web UI asset embedder (empty asset table unless you supply a build) |
| `diffusion/` | The diffusion runtime (models, samplers, tokenizers, loaders) and CLI |
| `quantizer/` | The quantizer library, and the `qz_*` GGUF block codec |
| `thirdparty/` | cpp-httplib, nlohmann/json, stb, miniaudio, subprocess.h, darts |

## In this section

| Document | What it covers |
| -------- | -------------- |
| [gk.md](gk.md) | The compute library: backends, scheduling, memory, debug switches |
| [quantizer.md](quantizer.md) | The `qz_*` codec and the encoder library |

The engine's own READMEs go further:

- `vendor/engine/README.md` — build options
- `vendor/engine/app/README.md` — the complete HTTP API (2000 lines)
- `vendor/engine/gk/README.md` — gk internals, benchmarks, test strategy
- `vendor/engine/diffusion/README.md` — diffusion runtime notes
- `vendor/engine/mtmd/README.md` — multimodal background

## Backends

GPU backends are opt-in and apply to the **whole engine** — server, diffusion
runtime and multimodal projectors all run on the one gk build.

| Engine option | `ggk` equivalent | Backend |
| ------------- | ---------------- | ------- |
| `-DGGUF_SERVER_CUDA=ON` | `GGK_CUDA=1` | NVIDIA CUDA |
| `-DGGUF_SERVER_HIP=ON` | `GGK_HIP=1` | AMD ROCm / HIP |
| `-DGGUF_SERVER_VULKAN=ON` | `GGK_VULKAN=1` | Vulkan |
| `-DGGUF_SERVER_METAL=ON` | `GGK_METAL=1` | Apple Metal (macOS default) |

Each forwards to the matching `GK_*` option.

Backends gk does **not** implement — and which are therefore simply absent,
not stubbed — include SYCL, CANN, OpenCL, WebGPU, RPC, zDNN, ZenDNN,
OpenVINO, Hexagon, MUSA, virtGPU and ET. The training/autograd half of the
old ggml API is stubbed: the `*_BACK` ops and training steps hold enum slots
for compatibility and abort if reached. Nothing in this tree calls them.

## Building standalone

The engine builds on its own, without Python:

```sh
cd vendor/engine
cmake -B build
cmake --build build -j
```

produces `build/bin/gguf-server`, `build/bin/diffusion` and
`build/bin/libgguf_quantizer.so`.

See [../build.md](../build.md) for the full option list.

## One constant worth knowing

`GGML_MAX_NAME` and `GK_MAX_NAME` are both fixed at **128** for the whole
tree. Diffusion weight names run past the historical 64-byte cap, and the
compat layer's `ggml_tensor` and gk's `gk_tensor` are layout-identical by
construction — so widening one without the other is exactly what makes the
static assert in `ggml-compat-impl.h` fire.

## License

MIT, inherited from llama.cpp and stable-diffusion.cpp. `gk/` carries its own
(also MIT) license, and third-party components under `thirdparty/` keep
theirs.
