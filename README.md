# ggk

One package for working with GGUF models locally: an OpenAI-compatible LLM
server, a diffusion image/video/audio generator and a GGUF metadata/tensor
editor with a built-in quantizer — three panels on one GUI, powered by one
unified engine compiled in a single build on top of **gk**, an independent
tensor library. There is no ggml anywhere in the tree.

[<img src="https://raw.githubusercontent.com/calcuis/gguf-connector/master/gguf.gif" width="128" height="128">]

Full documentation is in [`docs/`](docs/README.md) — installation, the CLI,
each panel's tabs and API, supported models, worked examples, the engine
internals, and troubleshooting.

## Install

```bash
pip install ggk
```

The build compiles the bundled engine (CPU by default, Metal on macOS).
GPU backends are opt-in at install time:

```bash
GGK_CUDA=1 pip install ggk     # NVIDIA
GGK_HIP=1 pip install ggk      # AMD ROCm
GGK_VULKAN=1 pip install ggk   # Vulkan
```

Each switch drives the whole engine — the server, the diffusion runtime and
the multimodal projectors all evaluate their graphs on the one gk build.

## Building a wheel

```bash
python -m build --wheel
```

`--wheel` is not optional for a GPU build. Plain `python -m build` builds an
sdist first and then compiles the wheel *from* it in a temporary directory, so
every run starts from scratch and a multi-hour CUDA build spends those hours
inside a directory the OS is free to sweep — on Windows that surfaces at the
very end as `FileNotFoundError: ...\wheel\scripts` from scikit-build-core's
packaging step, long after the compile succeeded. With `--wheel` the CMake
build directory stays at `build/{wheel_tag}` in the tree and rebuilds are
incremental.

On Windows a CUDA build needs MSVC, because that is the only host compiler
nvcc accepts there — run it from a `vcvars64` shell. Use Ninja rather than
`-G "Visual Studio 17 2022"`: scikit-build-core passes no `-j` on the
pyproject path and CMake's Visual Studio generator does not set `/MP`, so an
MSBuild-driven build compiles one file at a time.

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_ARGS=-G Ninja -DGGK_CUDA=ON
python -m build --wheel
```

The build works its own architecture list out from nvcc and the installed
GPUs. A wheel built on one machine for another should say what it targets:
`-DGK_CUDA_ARCHITECTURES="75;86;89;120"`.

## Run

```bash
ggk                 # unified GUI — Server / Diffuser / Editor panels
python -m ggk       # same thing
```

Each panel also runs on its own, exactly like the standalone
gguf-server / gguf-diffusion / gguf-editor packages did:

```bash
ggk server          # LLM server GUI
ggk diffuser        # image generation GUI
ggk editor          # GGUF editor GUI
```

And the engines are directly scriptable from the CLI:

```bash
ggk server engine -- --model model.gguf --port 8888
ggk diffuser engine -- -m sd.gguf -p "a lighthouse at dusk" -o out.png
ggk editor quantize -m in.gguf -o out-q4_k.gguf --type q4_k
ggk editor devices
```

## Which hardware it picked

Every engine prints the device list to stderr before it does anything else:

```
gk: found 2 devices
  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB | compute capability = 8.9 | SMs = 20 | shared memory = 99 KiB | built for = 89
  CPU: gk CPU backend, 32014 MiB | SIMD = AVX2 | AVX2 = 1 | FMA = 1 | F16C = 1 | F16_VEC = 1
```

If a GPU you expected is missing, the install was a CPU-only one (the GPU
switches above are opt-in at install time) or its driver was not found — either
way the run works, on the CPU, at CPU speed, which is otherwise indistinguishable
from a slow GPU. `GK_QUIET=1` suppresses the banner.

## Layout

```
vendor/engine/           the unified ggk engine (one CMake build)
  gk/                    the gk compute kernels (CPU + optional GPU backends)
  gk/compat/             the historical ggml C API, implemented on gk
  src/ common/ mtmd/     GGUF LLM runtime
  app/                   the gguf-server HTTP server
  diffusion/             diffusion runtime + CLI
  quantizer/             quantizer shared library (its own quant kernels)
src/ggk/                 the Python package
  server/ diffuser/ editor/   the three panels (backend + web frontend each)
  gui.py static/         the unified 3-panel GUI shell
```

Nothing above `gk/compat/` knows gk exists: the runtimes include the same
`ggml.h` / `ggml-backend.h` / `gguf.h` headers and call the same functions
they always did, while graph building, allocation, scheduling and the kernels
themselves are gk's. See `vendor/engine/README.md` for the engine's own build
options.

See [`docs/development.md`](docs/development.md) for the layout in detail and
[`docs/engine/README.md`](docs/engine/README.md) for the engine's architecture.

The editor's quantizer stays independent — its `qz_*` codec is compiled both
into the quantizer library the editor drives and into gk itself, so the
encoder and the runtimes' decoder can never disagree about a GGUF block.
