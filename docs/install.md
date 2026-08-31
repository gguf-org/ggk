# Installing ggk

## Requirements

| | |
| --- | --- |
| Python | 3.8 or newer |
| Compiler | C11 + C++17 (GCC, Clang, or MSVC) |
| CMake | 3.15 or newer |
| Build backend | `scikit-build-core` (pulled in automatically) |

`pip install` compiles the bundled engine from source — there are no
pre-built binary wheels, because the engine's kernels are built for the
backends you ask for. A CPU-only build takes minutes; a CUDA build can take
hours.

## CPU (and Metal on macOS)

```bash
pip install ggk
```

This produces a CPU build everywhere, plus the Metal backend on macOS, which
is on by default there.

## GPU builds

GPU backends are opt-in **at install time**. One switch drives the whole
engine: the LLM server, the diffusion runtime and the multimodal projectors
all evaluate their graphs on the same gk build.

```bash
GGK_CUDA=1 pip install ggk     # NVIDIA CUDA
GGK_HIP=1 pip install ggk      # AMD ROCm / HIP
GGK_VULKAN=1 pip install ggk   # Vulkan
GGK_METAL=1 pip install ggk    # Apple Metal (already the macOS default)
```

The same switches are readable as CMake defines if you prefer:

```bash
CMAKE_ARGS="-DGGK_CUDA=ON" pip install ggk
```

Every `GGK_*` option is read from the environment as well as from `-D`,
because `pip install` makes environment variables far easier to pass. See
[build.md](build.md) for the complete option list, including the `GK_*`
knobs that pass straight through to the compute library.

### What a GPU switch actually enables

| Switch | gk backend | Quantizer device kernels |
| ------ | ---------- | ------------------------ |
| `GGK_CUDA` | CUDA | CUDA (`QUANTIZER_CUDA` follows it) |
| `GGK_HIP` | HIP | HIP (`QUANTIZER_HIP` follows it) |
| `GGK_VULKAN` | Vulkan | — (CPU) |
| `GGK_METAL` | Metal | — (opt in separately with `QUANTIZER_METAL`) |

The quantizer's GPU path is its own small kernel set, separate from gk's.

## Building a wheel

```bash
python -m build --wheel
```

`--wheel` is **not optional** for a GPU build. Plain `python -m build` builds
an sdist first and then compiles the wheel *from* it in a temporary
directory, so every run starts from scratch and a multi-hour CUDA build
spends those hours inside a directory the OS is free to sweep. On Windows
that surfaces at the very end as:

```
FileNotFoundError: ...\wheel\scripts
```

from scikit-build-core's packaging step — long after the compile succeeded.
With `--wheel` the CMake build directory stays at `build/{wheel_tag}` in the
tree and rebuilds are incremental.

## Editable install

```bash
pip install -e .
```

The CMake install rules deliberately place the artifacts **twice** — into the
wheel and into the source tree — so an editable install finds them:

```
src/ggk/server/bin/gguf-server
src/ggk/diffuser/bin/diffusion
src/ggk/editor/lib/libgguf_quantizer.{so,dylib,dll}
```

## Platform notes

### Windows + CUDA

MSVC is the only host compiler `nvcc` accepts on Windows, so run the build
from a `vcvars64` shell. Use Ninja rather than the Visual Studio generator:
scikit-build-core passes no `-j` on the pyproject path and CMake's Visual
Studio generator does not set `/MP`, so an MSBuild-driven build compiles one
file at a time.

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_ARGS=-G Ninja -DGGK_CUDA=ON
python -m build --wheel
```

### Windows + MinGW

A MinGW-built binary would otherwise need `libstdc++-6.dll`,
`libgcc_s_seh-1.dll` and `libwinpthread-1.dll` from the toolchain at run
time. The wheel ships self-contained binaries and Python spawns them without
MSYS2 on `PATH`, so the MinGW runtime is linked in statically by default
(`GGK_MINGW_STATIC=ON`). gk runs its own thread pool rather than OpenMP, so
there is no libgomp import library to fight here.

`_WIN32_WINNT` is forced to `0x0A00` up front, because MinGW-w64's headers
otherwise pin it to a Windows 7-era value and cpp-httplib hard-errors.

### Windows + OpenSSL

OpenSSL is **off by default** (`GGK_OPENSSL=OFF`). It is only needed for the
engine's own HTTPS model downloads; the panels always hand the engine local
file paths, so the dependency buys nothing and costs portability.

If you do want it under MSVC, a MinGW/MSYS2 OpenSSL on `PATH` is deliberately
ignored — its headers cannot be compiled by MSVC. Install an MSVC build and
point CMake at it:

```bat
vcpkg install openssl:x64-windows
set CMAKE_ARGS=-DGGK_OPENSSL=ON -DOPENSSL_ROOT_DIR=C:\vcpkg\installed\x64-windows
```

### Targeting another machine

A wheel built on one machine for another should say what it targets. Left to
itself, a CUDA build works its architecture list out from `nvcc` and the
installed GPUs, and embeds PTX for the newest of them so an unlisted card
JITs rather than failing:

```bash
CMAKE_ARGS='-DGGK_CUDA=ON -DGK_NATIVE=OFF -DGK_CUDA_ARCHITECTURES="75;86;89;120"' \
    python -m build --wheel
```

`GK_NATIVE=OFF` also stops the CPU kernels from being compiled for the build
machine's exact SIMD level; pair it with an explicit `GK_ARCH_FLAGS`.

## Verifying the install

```bash
ggk --version
ggk editor devices          # quantizer devices compiled into this build
ggk diffuser engine -- --list-devices
ggk server engine -- --version
```

Every engine also prints its device list to stderr before doing anything
else:

```
gk: found 2 devices
  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB | compute capability = 8.9 | SMs = 20 | shared memory = 99 KiB | built for = 89
  CPU: gk CPU backend, 32014 MiB | SIMD = AVX2 | AVX2 = 1 | FMA = 1 | F16C = 1 | F16_VEC = 1
```

If a GPU you expected is missing, either the install was a CPU-only one (the
switches above are opt-in) or the driver was not found. Either way the run
works — on the CPU, at CPU speed, which is otherwise indistinguishable from a
slow GPU. `GK_QUIET=1` suppresses the banner.

## Uninstalling

```bash
pip uninstall ggk
```

Models, generated images and saved presets live outside the package: presets
and workflows are in the browser's `localStorage`, images wherever you chose
to write them.
