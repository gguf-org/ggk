# Hardware, backends and devices

## What backend am I actually running on?

Every engine prints its device list to stderr before it does anything else:

```
gk: found 2 devices
  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB | compute capability = 8.9 | SMs = 20 | shared memory = 99 KiB | built for = 89
  CPU: gk CPU backend, 32014 MiB | SIMD = AVX2 | AVX2 = 1 | FMA = 1 | F16C = 1 | F16_VEC = 1
```

The trailing pairs report what the **build** chose, not what the machine has:
the vector path the CPU kernels were compiled for, the architectures a CUDA
build carries code for. Those are exactly the questions behind a run that is
unexpectedly slow, and none of them can be seen from the outside.

If a GPU you expected is missing, one of two things happened:

1. the install was CPU-only — the GPU switches are opt-in at install time
   ([install.md](install.md)), or
2. the driver was not found, or the device was skipped for having no kernel
   image for its architecture.

Either way the run works, on the CPU, at CPU speed — which is otherwise
indistinguishable from a slow GPU. That is why the banner exists.

`GK_QUIET=1` turns it off.

## Two different device lists

This trips people up, so it is worth stating plainly.

| Source | What it is | Use it for |
| ------ | ---------- | ---------- |
| **The engine** (`--list-devices`, `/api/devices`) | The devices gk registered, in gk's order | Anything that names a device: `--backend`, `--main-gpu`, `--tensor-split` |
| **`nvidia-smi`** (`/api/hardware`, the Hardware tabs) | The system's view | Watching temperature, utilisation, free VRAM |

**They do not agree.** CUDA numbers its devices fastest-first by default, so
on a machine with an RTX 5090 and an RTX 4050 the engine's `cuda0` is the
5090 while `nvidia-smi` calls the 4050 GPU 0. Guessing gets a tensor split
backwards, silently.

Get the names from the engine:

```bash
ggk diffuser engine -- --list-devices
```

```json
[{"name": "cuda0", "description": "NVIDIA GeForce RTX 5090", "is_gpu": true,
  "memory_free": 30000000000, "memory_total": 34359738368},
 {"name": "cuda1", "description": "NVIDIA GeForce RTX 4050", "is_gpu": true, "…": "…"},
 {"name": "cpu",   "description": "gk CPU backend", "is_gpu": false}]
```

A device the build has no kernels for is not listed at all — the engine drops
it at discovery rather than failing partway through a run.

## Backend characteristics

| Backend | Platform | Notes |
| ------- | -------- | ----- |
| **CPU** | everywhere | Always present. AVX-512 / AVX2 / NEON / scalar, chosen at compile time — there is no runtime dispatch |
| **CUDA** | NVIDIA | The widest op coverage. Decodes every block format except the lattice families (IQ1/IQ2/IQ3), which fall back to the CPU |
| **HIP** | AMD ROCm | The same `.cu` sources, so the same coverage |
| **Metal** | Apple | The CUDA set minus fused attention, im2col and resampling, and minus the ternary and micro-scaling formats |
| **Vulkan** | cross-vendor | The Metal set minus group norm; destinations must be f32 |

An op a backend does not support runs on the CPU instead — a **fallback, not
a failure**. What that costs is a staged copy per boundary, which is why the
op sets are as wide as they are. If a model is unexpectedly slow on a GPU,
this is usually why.

## Splitting across GPUs

### LLM server

```bash
ggk server engine -- --model model.gguf \
    --n-gpu-layers 999 --main-gpu 0 --tensor-split 24,8
```

`--tensor-split` takes proportions, not counts. `24,8` puts three quarters of
the layers on device 0 and one quarter on device 1 — the shape you want when
device 0 has 24 GB and device 1 has 8 GB.

`--main-gpu` is where the non-split tensors and the KV cache live, so make it
the device with the most headroom.

The Server panel's **Recommended** button (Settings → Performance) derives both: the main GPU is the one
with the most free VRAM, and the split is the free-VRAM weights of every CUDA
device, when there are at least two usable ones.

### Diffusion

Per-module, which is finer-grained:

```bash
ggk diffuser engine -- \
    --backend "diffusion=cuda0&cuda1,te=cpu,vae=cpu" \
    --split-mode "diffusion=layer" …
```

| Module | What |
| ------ | ---- |
| `diffusion` | The main transformer/UNet |
| `te` | Text encoders |
| `clip-vision` | The CLIP vision tower |
| `vae` | The latent decoder |
| `control-net` | ControlNet |
| `photo-maker` | PhotoMaker |
| `upscaler` | The upscale model |

`&` joins devices for one module. `--split-mode` is `layer` (whole transformer
blocks per device — the default) or `row` (matmul rows split across devices,
**CUDA only**).

Text encoders and the VAE are used briefly and are good candidates for the
CPU when VRAM is tight; the diffusion model is used every step and is not.

## Fitting a model in memory

### LLM server

In rough order of what to try:

1. **Fewer GPU layers.** `--n-gpu-layers 20` offloads part of the model and
   runs the rest on the CPU.
2. **Smaller context.** `--ctx-size` dominates KV cache size, which is often
   larger than people expect at long contexts.
3. **Quantized KV cache.** `--cache-type-k q8_0 --cache-type-v q8_0` roughly
   halves it. Turn flash attention on with it.
4. **Smaller batches.** `--batch-size 512 --ubatch-size 256`.
5. **A smaller quantization.** `q4_k` instead of `q6_k`, and so on.
6. **`--no-mmap`** if the OS is thrashing the page cache; **`--mlock`** if
   the model fits and you want it pinned.

### Diffusion

1. `--diffusion-fa` — flash attention in the diffusion model.
2. `--offload-to-cpu` — weights live in RAM and stream to VRAM on use.
3. `--vae-tiling` (`--vae-tile-size`, `--vae-tile-overlap`) — the VAE decode
   is often the peak, not the diffusion.
4. A smaller quantization of the transformer.
5. `--max-vram 6 --stream-layers` — an explicit budget with residency and
   prefetch.
6. `--backend te=cpu,vae=cpu` — move what is used least onto the CPU.
7. `--temporal-tiling` for video.

## Threads

Both engines take `-t / --threads`, and `0` (or `-1`) auto-detects.

gk's pool runs one barrier per graph node, so a very small graph can be
faster on one thread than on many. Work splits by destination row, which
makes results **bit identical at any thread count** — so changing threads
changes speed, never output.

More threads than physical cores rarely helps and often hurts; SMT siblings
contend for the same vector units.

## Reproducibility

- **Same machine, same build, same seed → same output**, at any thread count.
  gk fixes the accumulation order per row and never lets it depend on the
  split.
- **Different backend → possibly different output.** The CPU kernel is the
  definition and the device kernel reproduces it as closely as the hardware
  allows, but not bit-for-bit in every case.
- **Different SIMD baseline → possibly different output**, for the same
  reason. `GK_NATIVE=OFF` with a fixed `GK_ARCH_FLAGS` is how you pin it.

## Monitoring

| Where | What |
| ----- | ---- |
| Server panel → **Hardware** | CPU, RAM, per-GPU VRAM/utilisation/temperature, live |
| Diffuser panel → **Hardware** | CPU count, RAM, NVIDIA GPUs |
| `GET /metrics` on the LLM server | Prometheus metrics (needs `--metrics`) |
| `GET /slots` on the LLM server | Per-slot state and context use |
| `nvidia-smi -l 1` | The ground truth for NVIDIA |

## Environment variables

| Variable | Effect |
| -------- | ------ |
| `GK_QUIET=1` | Suppress the device banner |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs are visible. The server panel sets it to `-1` for the child when GPU layers is 0 |
| `GK_NODE_HASH`, `GK_ALLOC_NO_REUSE`, `GK_ALLOC_TRACE`, `GK_FA_*`, … | gk debug switches — [engine/gk.md](engine/gk.md#debug-switches) |
