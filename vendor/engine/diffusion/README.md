# diffusion

A standalone, self-contained diffusion image/video generation engine in C/C++.

- **No external dependencies, and no ggml.** The graphs the engine builds are
  evaluated by `gk`, the compute library at the engine root (`../gk/`), through
  the ggml-API compatibility layer in `../gk/compat/` — the same one the LLM
  server and the multimodal projectors link against, compiled once for all
  three. The only other third-party code is a handful of single-file headers in
  `thirdparty/` (stb image I/O, nlohmann json, darts). There is no ggml
  submodule, no libwebp, no libwebm, no zip.
- **GGUF and safetensors model loading.** Conversion and quantization are out
  of scope — use the separate `quantizer` project to prepare GGUF files.
- **One executable:** `diffusion` (plus the `diffusion` shared library).

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The binary is written to `build/bin/diffusion`.

### GPU backends

| Backend | Configure flags |
|---------|-----------------|
| CUDA    | `-DSD_CUDA=ON` |
| ROCm    | `-DSD_HIPBLAS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DAMDGPU_TARGETS=gfx1100` |
| Metal   | `-DSD_METAL=ON` |
| Vulkan  | `-DSD_VULKAN=ON` |

Example (CUDA):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
cmake --build build -j
```

Example (METAL):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON
cmake --build build -j
```
* you might need to build or sign it yourself if you don't know how to handle the warning from pre-built Releases

### Other options

- `-DSD_BUILD_SHARED_LIBS=OFF` — build the diffusion library statically.
- `-DSD_BUILD_CLI=OFF` — build only the library.

## Usage

```sh
./build/bin/diffusion \
    --diffusion-model model.gguf \
    --llm text_encoder.gguf \
    --vae vae.gguf \
    -p "a watercolor fox" \
    --steps 8 -W 512 -H 512 \
    -o out.png
```

Run `./build/bin/diffusion --help` for the full option list. Supported run
modes: `img_gen` (default), `vid_gen`, `upscale`, `metadata`.

Output formats: PNG / JPG / BMP / TGA (video modes write image sequences or
AVI).

## Layout

```
include/gk-diffuser.h        C API of the diffusion library
src/                         engine (models, samplers, tokenizers, loaders)
cli/                         the diffusion command-line app
thirdparty/                  single-file headers (stb, json, darts)
```

The compute kernels are not here: `../gk/` (CPU/CUDA/HIP/Metal/Vulkan) and the
ggml compatibility layer `../gk/compat/` are shared with the rest of the
engine, so backend selection is engine-wide rather than per-runtime.

## Removed compared to the original diffusion.cpp project

- ggml as an external/vendored repo dependency (only the needed kernel tree is kept)
- convert / quantize modes (`--mode convert`, imatrix collection) — use `quantizer`
- PyTorch checkpoint loading (`.ckpt`/`.pt`/`.pth` pickle + zip)
- WebP image and WebM video output (libwebp / libwebm)


## Reference

Original diffusion.cpp project: [diffusion.cpp](https://github.com/mochiyaki/diffusion.cpp); thanks all contributors in the Community🤖, especially leejet - stable-diffusion.cpp (MIT License), ggerganov - ggml (MIT License) and etc., to make the based code open sourced! Really appreciate!♥️

## License

This project is licensed under the MIT License. See `LICENSE`.
