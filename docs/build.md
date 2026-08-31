# Building from source

`pip install ggk` already compiles the engine — this page is for building it
yourself, changing options, or working on the C/C++ side.

## The two builds

| | |
| --- | --- |
| **Package build** | Root `CMakeLists.txt`. Adds the engine tree as a subdirectory, then installs the three artifacts *into the Python package*. Driven by scikit-build-core through `pip` / `python -m build`. |
| **Engine build** | `vendor/engine/CMakeLists.txt`. Standalone; produces `build/bin/{gguf-server,diffusion,libgguf_quantizer}`. No Python involved. |

## Package build

```bash
pip install -e .              # editable
python -m build --wheel       # a wheel
```

The root `CMakeLists.txt`:

1. Locates the engine tree — `vendor/engine`, or `-DGGK_ENGINE_DIR=…`, or the
   `GGK_ENGINE_DIR` environment variable.
2. Forces `BUILD_SHARED_LIBS=OFF`, so gk, the compat layer, llama, mtmd and
   the diffusion library link straight into the executables. The wheel then
   ships standalone binaries with no rpath or DLL handling. The quantizer
   stays shared by design — ctypes needs to `dlopen` it.
3. Adds the engine subdirectory `EXCLUDE_FROM_ALL`, so only the three shipped
   artifacts and their dependencies are built.
4. Installs into the wheel **and** into the source tree, so editable installs
   find the artifacts:

```
src/ggk/server/bin/gguf-server
src/ggk/diffuser/bin/diffusion
src/ggk/editor/lib/libgguf_quantizer.{so,dylib,dll}
```

### Options

Every one is readable from the environment as well as from `-D`, because
`pip install` makes environment variables far easier to pass:

```bash
GGK_CUDA=1 pip install .
CMAKE_ARGS="-DGGK_CUDA=ON" pip install .
```

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `GGK_BUILD` | `ON` | Build the engine at all. `OFF` gives a Python-only install with no engines |
| `GGK_CUDA` | `OFF` | NVIDIA CUDA backend (also sets `QUANTIZER_CUDA`) |
| `GGK_HIP` | `OFF` | AMD ROCm/HIP backend (also sets `QUANTIZER_HIP`) |
| `GGK_VULKAN` | `OFF` | Vulkan backend |
| `GGK_METAL` | `ON` on macOS | Apple Metal backend |
| `GGK_OPENSSL` | `OFF` | Link the server engine against OpenSSL for HTTPS model downloads |
| `GGK_SUBPROCESS` | `ON` | Router mode (`--models-dir`) — the server spawns child servers |
| `GGK_MINGW_STATIC` | `ON` | Statically link the MinGW runtime (Windows, non-MSVC) |
| `GGK_ENGINE_DIR` | `vendor/engine` | Engine source tree |
| `GGUF_SERVER_UI_DIR` | unset | Static web assets to embed into the server engine |

Anything gk understands passes straight through as `-DGK_*`:

| Option | Meaning |
| ------ | ------- |
| `GK_NATIVE` | Compile CPU kernels for the building machine (default `ON`) |
| `GK_ARCH_FLAGS` | Explicit CPU baseline when `GK_NATIVE=OFF` |
| `GK_CUDA_ARCHITECTURES` | e.g. `"75;86;89;120"` |
| `GK_HIP_ARCHITECTURES` | e.g. `"gfx1100"` |

A build meant to run on another machine wants `-DGK_NATIVE=OFF`, an explicit
`GK_ARCH_FLAGS`, and an explicit `GK_CUDA_ARCHITECTURES`. See
[engine/gk.md § CUDA architectures](engine/gk.md#cuda-architectures) for why
CMake's `native` is the wrong tool here.

### Why `--wheel` matters

Plain `python -m build` builds an sdist first, then compiles the wheel *from*
it in a temp directory. Every run starts from scratch, and a multi-hour CUDA
build spends those hours somewhere the OS may sweep. On Windows that surfaces
at the very end as `FileNotFoundError: ...\wheel\scripts`, long after the
compile succeeded.

`--wheel` keeps the CMake build directory at `build/{wheel_tag}` in the tree,
and rebuilds are incremental.

## Engine build

```sh
cd vendor/engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

| Artifact | What |
| -------- | ---- |
| `build/bin/gguf-server` | The LLM server |
| `build/bin/diffusion` | The image/video generation CLI |
| `build/bin/libgguf_quantizer.so` | The quantizer library |

### Engine options

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `GGUF_SERVER_CUDA` | `OFF` | NVIDIA CUDA |
| `GGUF_SERVER_HIP` | `OFF` | AMD ROCm/HIP |
| `GGUF_SERVER_VULKAN` | `OFF` | Vulkan |
| `GGUF_SERVER_METAL` | `ON` on macOS | Apple Metal |
| `GGUF_SERVER_OPENSSL` | `ON` | HTTPS model downloads |
| `GGUF_SERVER_SUBPROCESS` | `ON` | Router mode |
| `GGUF_SERVER_INSTALL` | | Install rules |
| `GGUF_SERVER_ALL_WARNINGS` | | Extra warnings |
| `GGUF_SERVER_SANITIZE_*` | | Sanitizer builds |
| `GGUF_SERVER_UI_DIR` | `ui/dist` | Web assets to embed |
| `SD_BUILD_CLI` | `ON` | Build the diffusion CLI as well as the library |
| `QUANTIZER_CUDA` / `_HIP` / `_METAL` | | The quantizer's own device kernels |

Note the different `GGUF_SERVER_OPENSSL` default: `ON` for a standalone
engine build, forced `OFF` by the `ggk` package build, because the panels
always hand the engine local file paths.

Each `GGUF_SERVER_*` backend option forwards to the matching `GK_*`.

### The diffusion runtime on its own

```sh
cd vendor/engine/diffusion
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
cmake --build build -j
```

| Backend | Flags |
| ------- | ----- |
| CUDA | `-DSD_CUDA=ON` |
| ROCm | `-DSD_HIPBLAS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DAMDGPU_TARGETS=gfx1100` |
| Metal | `-DSD_METAL=ON` |
| Vulkan | `-DSD_VULKAN=ON` |

`-DSD_BUILD_SHARED_LIBS=OFF` builds the library statically;
`-DSD_BUILD_CLI=OFF` builds only the library.

### gk on its own

```sh
cd vendor/engine/gk
cmake -B build -DGK_CUDA=ON
cmake --build build -j
./build/tests/test-foundation
./build/tests/bench-gk
```

## Platform specifics

### Windows + MSVC + CUDA

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_ARGS=-G Ninja -DGGK_CUDA=ON
python -m build --wheel
```

MSVC is the only host compiler `nvcc` accepts on Windows. Use **Ninja**, not
the Visual Studio generator: scikit-build-core passes no `-j` on the pyproject
path and CMake's VS generator does not set `/MP`, so an MSBuild-driven build
compiles one file at a time.

### Windows + MinGW

The build sets `_WIN32_WINNT=0x0A00` up front, because MinGW-w64's headers
otherwise pin it to a Windows 7-era value, cpp-httplib hard-errors, and
`::CreateFile2` goes undeclared.

`GGK_MINGW_STATIC=ON` (the default) links `-static-libgcc -static-libstdc++
-static`, so the binaries do not need `libstdc++-6.dll`,
`libgcc_s_seh-1.dll` or `libwinpthread-1.dll` at run time. gk runs its own
thread pool rather than OpenMP, so there is no libgomp import library to
fight — unlike the ggml build this tree used to carry.

### Windows + OpenSSL

Under MSVC, a MinGW/MSYS2 OpenSSL on `PATH` is deliberately ignored — its
headers cannot be compiled by MSVC. Install an MSVC build and point CMake at
it, or leave `GGK_OPENSSL` off:

```bat
vcpkg install openssl:x64-windows
set CMAKE_ARGS=-DGGK_OPENSSL=ON -DOPENSSL_ROOT_DIR=C:\vcpkg\installed\x64-windows
```

### macOS

Metal is on by default. The quantizer's Metal kernels are separate and stay
opt-in:

```bash
CMAKE_ARGS="-DQUANTIZER_METAL=ON" pip install -e .
```

## Iterating on the C/C++ side

The engine is a normal CMake subproject, so the fastest loop is to build it
directly and copy the artifact into place:

```sh
cd vendor/engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target gguf-server
cp build/bin/gguf-server ../../src/ggk/server/bin/
```

For the Python side, nothing needs rebuilding: an editable install picks up
edits to `src/ggk/**` immediately, and the static frontends are read from
disk on every request with `Cache-Control: no-store`.

## Verifying a build

```bash
ggk --version
ggk server engine -- --version
ggk diffuser engine -- --version
ggk diffuser engine -- --list-devices
ggk editor devices
```

Each engine prints its gk device banner to stderr on startup — that is the
quickest confirmation that a backend actually got compiled in. See
[hardware.md](hardware.md).
