# ggk engine

The unified engine behind the `ggk` Python package: an OpenAI-compatible HTTP
server for GGUF language models, an image/video diffusion CLI, and a GGUF
quantizer library — all compiled from this one tree, in one build, against one
copy of the compute kernels.

There is **no vendored `llama.cpp` checkout and no `ggml` at all**: the graphs
the runtimes build are evaluated by **gk**, an independent tensor library, and
everything — kernels, GGUF runtime, diffusion engine, HTTP layer — is compiled
from the sources in this directory.

```sh
cmake -B build
cmake --build build -j
```

produces three artifacts in `build/bin/`:

| Artifact               | What it is                                                |
| ---------------------- | --------------------------------------------------------- |
| `gguf-server`          | the LLM server (`--model model.gguf` -> http://127.0.0.1:8888) |
| `diffusion`            | the image/video generation CLI                             |
| `libgguf_quantizer.so` | the quantizer, loaded by the editor panel via ctypes       |

## Layout

| Directory      | What it holds                                                            |
| -------------- | ------------------------------------------------------------------------ |
| `gk/`          | the gk compute kernels: CPU plus optional CUDA / HIP / Metal / Vulkan     |
| `gk/compat/`   | the historical `ggml` C API implemented on gk — what the runtimes link against |
| `src/`         | the GGUF model runtime (`libllama`): loader, graphs, KV cache, sampling    |
| `include/`     | the runtime's public headers (`llama.h`, `llama-cpp.h`)                    |
| `common/`      | argument parsing, chat templates (jinja), sampling presets, logging, downloads |
| `mtmd/`        | multimodal projectors — vision and audio inputs                           |
| `app/`         | the HTTP server itself: routes, task queue, slots, streaming, router mode |
| `ui/`          | web UI asset embedder (empty asset table unless you supply a build)       |
| `diffusion/`   | the diffusion runtime (models, samplers, tokenizers, loaders) and its CLI  |
| `quantizer/`   | the quantizer library, and the `qz_*` GGUF block codec gk compiles in      |
| `thirdparty/`  | cpp-httplib, nlohmann/json, stb, miniaudio, subprocess.h, darts           |

Nothing above `gk/compat/` knows gk exists: `src/`, `mtmd/`, `common/` and
`diffusion/` include the `ggml.h` / `ggml-backend.h` / `gguf.h` headers in
`gk/compat/include/` and call the same functions they always did. The compat
layer is a thin translation — graph building, allocation, scheduling and the
kernels themselves are gk's.

gk implements the backends this tree ships (CPU, CUDA, HIP, Metal, Vulkan);
SYCL, CANN, OpenCL, WebGPU, RPC, zDNN, ZenDNN, OpenVINO, Hexagon, MUSA, virtGPU
and ET have no gk counterpart and are simply absent. The training / autograd
half of the old `ggml` API is stubbed out — nothing here calls it.

### The quantizer, twice

`quantizer/src/kernels/qz_*` is the one place in this tree where the GGUF
on-disk block layouts are implemented, and it is compiled into two different
things:

* `libgguf_quantizer`, the encoder the editor panel drives — its own library,
  sharing nothing else with the runtimes, because encoding is a different job
  from inference; and
* `gk` itself, which links those same sources for the decode side.

Compiling them in rather than duplicating them is what keeps the engine and the
quantizer from ever disagreeing about a block.

## Build options

GPU backends are opt-in and apply to the whole engine — the server, the
diffusion runtime and the multimodal projectors all run on the one gk build:

| Option                    | Backend                              |
| ------------------------- | ------------------------------------ |
| `-DGGUF_SERVER_CUDA=ON`   | NVIDIA CUDA                          |
| `-DGGUF_SERVER_HIP=ON`    | AMD ROCm/HIP                         |
| `-DGGUF_SERVER_VULKAN=ON` | Vulkan                               |
| `-DGGUF_SERVER_METAL=ON`  | Apple Metal (on by default on macOS) |

Each forwards to the matching `GK_*` option.

Other options: `GGUF_SERVER_OPENSSL` (HTTPS model downloads, ON),
`GGUF_SERVER_SUBPROCESS` (router mode, ON), `GGUF_SERVER_INSTALL`,
`GGUF_SERVER_ALL_WARNINGS`, `GGUF_SERVER_SANITIZE_*`, `SD_BUILD_CLI`,
`QUANTIZER_CUDA` / `QUANTIZER_HIP` / `QUANTIZER_METAL` (the quantizer's own
device kernels, separate from gk's). gk's knobs (`GK_NATIVE`, `GK_ARCH_FLAGS`,
`GK_CUDA_ARCHITECTURES`, `GK_HIP_ARCHITECTURES`) can be passed straight
through — a build meant to run on another machine wants `-DGK_NATIVE=OFF`, an
explicit `GK_ARCH_FLAGS`, and an explicit `GK_CUDA_ARCHITECTURES`. Left to
itself a CUDA build targets the cards it can see plus PTX for the newest of
them; see `gk/README.md` for why that is not CMake's `native`.

`GGML_MAX_NAME` and `GK_MAX_NAME` are both fixed at 128 for the whole tree.
Diffusion weight names run past the historical 64-byte cap, and the compat
layer's `ggml_tensor` and gk's `gk_tensor` are layout-identical by construction
(a static assert in `ggml-compat-impl.h` checks it), so widening one without
the other is what makes that assert fire.

On Windows with MSVC, a MinGW/MSYS2 OpenSSL on `PATH` is deliberately ignored —
its headers cannot be compiled by MSVC. For HTTPS model downloads install an
MSVC build (`vcpkg install openssl:x64-windows`) and pass
`-DOPENSSL_ROOT_DIR=<prefix>`, or build without it via `-DGGUF_SERVER_OPENSSL=OFF`.

## Usage

```sh
# LLM server: local model, default host/port -> http://127.0.0.1:8888
./build/bin/gguf-server --model model.gguf

# multimodal: model + projector
./build/bin/gguf-server --model model.gguf --mmproj mmproj.gguf

# router mode: serve several models, loading them on demand
./build/bin/gguf-server --models /path/to/gguf-dir --models-autoload

# image generation
./build/bin/diffusion --diffusion-model model.gguf --vae vae.gguf \
    -p "a red cube on a white table" -o out.png
```

Defaults that differ from upstream `llama-server`: the binary is `gguf-server`
and the listen port is **8888** (`--port` / `LLAMA_ARG_PORT` still override it).
Environment variables keep their `LLAMA_ARG_*` names.

`app/README.md` is the full HTTP API reference: `/completion`,
`/v1/chat/completions`, `/v1/embeddings`, `/rerank`, `/infill`, `/props`,
`/slots`, `/metrics`, and the rest.

## Web UI

`ui/` embeds static web assets into the `gguf-server` binary. This tree ships no
asset bundle, so a plain build is API-only and `/` returns 404 — which is what
the `ggk` package wants, since it serves its own GUI. To get one anyway:

* build one and point the embedder at it:
  `cmake -B build -DGGUF_SERVER_UI_DIR=/path/to/dist` (any directory with an
  `index.html`; all files under it are embedded), or drop it at `ui/dist`;
* or serve it from disk at runtime with `gguf-server --path /path/to/dist`.

## License

MIT, inherited from llama.cpp and stable-diffusion.cpp — see `LICENSE`. `gk/`
carries its own (also MIT) `LICENSE`, and third-party components under
`thirdparty/` keep theirs.
