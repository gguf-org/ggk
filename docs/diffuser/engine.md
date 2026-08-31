# The `diffusion` engine

A single C/C++ executable built from `vendor/engine/diffusion/` and installed
at `src/ggk/diffuser/bin/diffusion`. It loads GGUF and safetensors weights;
conversion and quantization are out of scope — use the
[Editor panel](../editor/README.md) for those.

```bash
ggk diffuser engine -- --help
ggk diffuser engine -- --version
ggk diffuser engine -- --list-devices     # JSON, what --backend accepts
```

`--version` and `--list-devices` must be the **first** argument.

## Run modes

```
-M, --mode  img_gen | vid_gen | upscale | metadata     (default: img_gen)
```

| Mode | Requires | Produces |
| ---- | -------- | -------- |
| `img_gen` | a diffusion model, `-o` | one image, or a batch |
| `vid_gen` | a video model, `-o` | an image sequence or AVI |
| `upscale` | `--upscale-model`, `-i`, `-o` | an upscaled image |
| `metadata` | `--image` | the generation metadata embedded in an image |

## Model files

| Flag | What |
| ---- | ---- |
| `-m, --model` | Full/all-in-one model |
| `--diffusion-model` | Standalone diffusion transformer or UNet |
| `--high-noise-diffusion-model` | High-noise expert (Wan 2.2 MoE) |
| `--uncond-diffusion-model` | Unconditional model (Ideogram4 CFG) |
| `--vae` | Standalone VAE |
| `--vae-format` | Latent format override: `auto`, `flux`, `sd3`, `flux2` |
| `--audio-vae` | Audio VAE (LTXAV, MiniMax H3) |
| `--taesd`, `--tae` | Tiny AutoEncoder — fast, low-quality decode |
| `--control-net` | ControlNet |
| `--upscale-model` | ESRGAN-family upscaler |
| `--photo-maker` | PhotoMaker |
| `--pulid-weights` | PuLID Flux weights |
| `--embeddings-connectors` | LTXAV embeddings connectors |
| `--embd-dir` | Textual-inversion embeddings directory |
| `--lora-model-dir` | LoRA directory (referenced from the prompt) |
| `--hires-upscalers-dir` | Upscaler directory for hires fix |
| `--tokenizer-pack` | Replacement tokenizer data dir (env `SD_TOKENIZER_PACK`) |

## Text encoders

| Flag | What |
| ---- | ---- |
| `--clip_l` | CLIP-L |
| `--clip_g` | CLIP-G |
| `--clip_vision` | CLIP vision tower |
| `--t5xxl` | T5-XXL |
| `--llm` | LLM text encoder (Qwen2.5-VL, Mistral Small 3.2, …) |
| `--llm_vision` | The LLM's ViT |
| `--llm-adapter` | Qwen3 → T5-XXL bridge adapter |
| `--qwen2vl`, `--qwen2vl_vision` | Deprecated aliases of `--llm` / `--llm_vision` |

## Prompt and inputs

| Flag | What |
| ---- | ---- |
| `-p, --prompt` | The prompt |
| `-n, --negative-prompt` | The negative prompt |
| `--prompt-file`, `--negative-prompt-file` | Read prompts from files |
| `--lyrics` | Lyrics for music models (MiniMax Music 3); empty = instrumental |
| `-i, --init-img` | img2img source |
| `--end-img` | End frame, for first/last-frame video |
| `--mask` | Inpainting mask |
| `--control-image` | ControlNet conditioning image |
| `--control-video` | Directory of control frames, lexicographically ordered |
| `-r, --ref-image` | Reference image; repeatable (Flux Kontext, edit models) |
| `--increase-ref-index` | Number references from 1 in listed order |
| `--disable-auto-resize-ref-image` | Keep references at native size |
| `--pm-id-images-dir`, `--pm-id-embed-path` | PhotoMaker identity inputs |
| `--pulid-id-embedding`, `--pulid-id-weight` | PuLID identity injection |

## Sampling

| Flag | Default | What |
| ---- | ------- | ---- |
| `--steps` | 20 | Sample steps |
| `--sampling-method` | model-specific | `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`, `res_multistep`, `res_2s`, `er_sde`, `euler_cfg_pp`, `euler_a_cfg_pp`, `euler_ge` |
| `--scheduler` | model-specific | `discrete` (alias `normal`), `karras`, `exponential`, `ays`, `gits`, `smoothstep`, `sgm_uniform`, `simple`, `kl_optimal`, `lcm`, `bong_tangent`, `ltx2`, `logit_normal`, `flux2`, `flux`, `beta` |
| `--sigmas` | — | Explicit comma-separated sigmas, e.g. `14.61,7.8,3.5,0.0` |
| `-s, --seed` | 42 | `< 0` = random |
| `--rng` | `cuda` | RNG: `std_default`, `cuda`, `cpu` |
| `--sampler-rng` | = `--rng` | Sampler RNG |
| `--eta` | method-specific | Noise multiplier |
| `--prediction` | auto | `eps`, `v`, `edm_v`, `sd3_flow`, `flux_flow`, `sefi_flow` |
| `--extra-sample-args` | — | `key=value` list; see below |

> **The flag is `--scheduler`, not `--schedule`.** The argument parser matches
> exact names only — there are no aliases and no prefix matching, and an
> unrecognised flag makes the engine print usage and exit non-zero. The one
> spelling alias that does exist is a *value*, not a flag: `normal` is
> accepted as a synonym for `discrete`.

`--extra-sample-args` takes method-specific keys:

| Method | Keys |
| ------ | ---- |
| CFG | `guidance_schedule` |
| APG | `apg_eta`, `apg_momentum`, `apg_norm_threshold`, `apg_norm_threshold_smoothing` |
| SLG | `slg_uncond` |
| `lcm` | `noise_clip_std`, `noise_scale_start`, `noise_scale_end` |
| `flux` | `base_shift`, `max_shift` |
| `ltx2` | `max_shift`, `base_shift`, `stretch`, `terminal` |
| `euler_ge` | `gamma` |
| `logit_normal` | `mu`, `std`, `logsnr_min`, `logsnr_max`, `resolution_aware` |

## Guidance

| Flag | Default | What |
| ---- | ------- | ---- |
| `--cfg-scale` | model-specific (usually 7.0) | Classifier-free guidance |
| `--img-cfg-scale` | = `--cfg-scale` | Image guidance for inpaint/edit models |
| `--guidance` | 3.5 | Distilled guidance for models with a guidance input |
| `--slg-scale` | 0 | Skip-layer guidance, DiT only; 2.5 suits SD3.5 medium |
| `--skip-layers` | `[7,8,9]` | Layers skipped during SLG steps |
| `--skip-layer-start` / `--skip-layer-end` | 0.01 / 0.2 | SLG window |
| `--flow-shift` | auto | Shift for flow models (SD3.x, Wan) |
| `--timestep-shift` | 0 | NitroFusion; ~250 for NitroSD-Realism, ~500 for Vibrant |
| `--clip-skip` | -1 | 1 skips none, 2 skips one layer |
| `--strength` | 0.75 | img2img denoising strength |
| `--control-strength` | 0.9 | ControlNet strength; 1.0 destroys the init image |
| `--vace-strength` | — | Wan VACE strength |
| `--pm-style-strength` | — | PhotoMaker style strength |

Every `--high-noise-*` twin (`--high-noise-steps`, `--high-noise-cfg-scale`,
`--high-noise-guidance`, `--high-noise-sampling-method`,
`--high-noise-slg-scale`, `--high-noise-eta`, `--high-noise-skip-layer*`)
applies to the high-noise expert of a two-stage MoE model.
`--moe-boundary` (default 0.875) sets the timestep handover, and only takes
effect when `--high-noise-steps` is `-1`.

## Size, batching, video

| Flag | Default | What |
| ---- | ------- | ---- |
| `-W, --width` / `-H, --height` | model-specific, usually 512 | Output size |
| `-b, --batch-count` | 1 | Images per run |
| `--video-frames` | 1 | Frames for video models |
| `--fps` | 24 | Output frame rate |
| `--audio-duration` | 60 | Seconds of audio (ACE-Step and friends) |
| `--qwen-image-layers` | 3 | Qwen-Image-Layered layer count; outputs = layers + 1 |

## Backends, devices and memory

| Flag | What |
| ---- | ---- |
| `--backend` | Runtime device per module: `cpu`, or `clip=cpu,vae=cuda0,diffusion=vulkan0`. Multiple devices with `&`: `diffusion=cuda0&cuda1` |
| `--split-mode` | `layer` (default) or `row` (CUDA only); one value or per-module |
| `--params-backend` | Where parameters live: `disk`, `cpu`, or per-module |
| `--max-vram` | VRAM budget in GiB for graph-cut execution. One value or `cuda0=6,vulkan0=4`. `0` disables; negative auto-detects free VRAM sparing that much |
| `--stream-layers` | Residency + prefetch streaming on top of `--max-vram` |
| `--eager-load` | Load all parameters at model-load time instead of lazily |
| `--offload-to-cpu` | Keep weights in RAM, move to VRAM on demand |
| `--mmap` | Memory-map the model file |
| `--rpc-servers` | Offload to RPC servers, `host:port,host:port` |
| `-t, --threads` | CPU threads; `<= 0` uses physical core count |

Deprecated shorthands, still accepted: `--clip-on-cpu` (`--backend te=cpu`),
`--vae-on-cpu` (`--backend vae=cpu`), `--control-net-cpu`
(`--backend controlnet=cpu`).

## Attention, convolution, precision

| Flag | What |
| ---- | ---- |
| `--fa` | Flash attention everywhere |
| `--diffusion-fa` | Flash attention in the diffusion model only |
| `--diffusion-conv-direct` | Direct conv2d in the diffusion model |
| `--vae-conv-direct` | Direct conv2d in the VAE |
| `--circular` | Circular padding for convolutions (tileable output) |
| `--circularx`, `--circulary` | Circular RoPE wrapping on one axis only |
| `--type` | On-the-fly weight type, e.g. `f16`, `q8_0`, `q4_k` |
| `--tensor-type-rules` | Per-pattern types, e.g. `^vae\.=f16,model\.=q8_0` |
| `--force-sdxl-vae-conv-scale` | Force conv scale on the SDXL VAE |

## VAE tiling

| Flag | Default | What |
| ---- | ------- | ---- |
| `--vae-tiling` | off | Decode in tiles to cut memory |
| `--vae-tile-size` | `32x32` | Tile size |
| `--vae-relative-tile-size` | — | `XxY`; fraction of image if < 1, tiles per dim if ≥ 1 |
| `--vae-tile-overlap` | 0.5 | Overlap as a fraction of tile size |
| `--temporal-tiling` | off | Temporal tiling for the LTX video VAE |
| `--extra-tiling-args` | — | `temporal_tile_frames` (4), `temporal_tile_overlap` (1) |

## Hires fix

| Flag | Default | What |
| ---- | ------- | ---- |
| `--hires` | off | Enable the second pass |
| `--hires-scale` | 2.0 | Scale when no explicit target size |
| `--hires-width` / `--hires-height` | 0 | Explicit target; 0 uses the scale |
| `--hires-steps` | 0 | Second-pass steps; 0 reuses `--steps` |
| `--hires-denoising-strength` | 0.7 | Second-pass strength |
| `--hires-sigmas` | — | Explicit second-pass sigmas |
| `--hires-upscaler` | `Latent` | `Lanczos`, `Nearest`, `Latent`, `Latent (nearest)`, `Latent (nearest-exact)`, `Latent (antialiased)`, `Latent (bicubic)`, `Latent (bicubic antialiased)`, or a model name under `--hires-upscalers-dir` |
| `--hires-upscale-tile-size` | 128 | Tile size for model-backed upscalers |

## Upscaling

| Flag | Default | What |
| ---- | ------- | ---- |
| `--upscale-repeats` | 1 | Run the ESRGAN upscaler N times |
| `--upscale-tile-size` | 128 | Tile size |

## Step caching

`--cache-mode` trades a little quality for a large speedup by reusing
transformer output across adjacent steps.

| Mode | For |
| ---- | --- |
| `easycache` | DiT models |
| `ucache` | UNET models |
| `dbcache` / `taylorseer` / `cache-dit` | DiT block-level caching |
| `spectrum` | UNET/DiT Chebyshev + Taylor forecasting |

`--cache-option` takes `key=value` pairs, comma-separated:

| Mode | Keys |
| ---- | ---- |
| `easycache`, `ucache` | `threshold`, `start`, `end`, `decay`, `relative`, `reset` |
| `dbcache`, `taylorseer`, `cache-dit` | `Fn`, `Bn`, `threshold`, `warmup` |
| `spectrum` | `w`, `m`, `lam`, `window`, `flex`, `warmup`, `stop` |

`--scm-mask` gives cache-dit an explicit per-step 0/1 compute mask;
`--scm-policy` is `dynamic` (default) or `static`.

## LoRA

LoRAs are referenced from the prompt (`<lora:name:weight>`), resolved against
`--lora-model-dir`.

| Flag | Default | What |
| ---- | ------- | ---- |
| `--lora-apply-mode` | `auto` | `auto`, `immediately`, `at_runtime` |

`immediately` folds the LoRA into the weights — faster inference, sometimes
lower memory, but precision and compatibility issues with quantized
parameters. `at_runtime` applies during evaluation, and is exactly the
opposite trade. `auto` picks `at_runtime` when any parameter is quantized.

## Model-specific

| Flag | For |
| ---- | --- |
| `--chroma-disable-dit-mask`, `--chroma-enable-t5-mask`, `--chroma-t5-mask-pad` | Chroma |
| `--qwen-image-zero-cond-t`, `--qwen-image-layers` | Qwen-Image |
| `--moe-boundary` | Wan 2.2 MoE |
| `--vace-strength` | Wan VACE |
| `--force-sdxl-vae-conv-scale` | SDXL |

## Output and preview

| Flag | Default | What |
| ---- | ------- | ---- |
| `-o` | required | Output path (extension picks PNG/JPG/BMP/TGA) |
| `--disable-image-metadata` | off | Don't embed generation metadata |
| `--output-begin-idx` | 0 or 1 | Start index for an image sequence (`-o out_%03d.png`) |
| `--preview` | `none` | Live preview method: `none`, `proj`, `tae`, `vae` |
| `--preview-path` | `./preview.png` | Where previews are written; `.avi` for multi-frame |
| `--preview-interval` | 1 | Denoising steps between preview updates |
| `--taesd-preview-only` | off | Use TAESD for previews but not for the final decode |
| `--preview-noisy` | off | Preview the noisy input rather than the denoised output |
| `--canny` | off | Apply the canny edge-detection preprocessor |
| `--color` | off | Colour the log tags by level |
| `-v` | off | Verbose logging |

In `--mode metadata`: `--image` names the file, `--metadata-format` is `text`
or `json`, and `--metadata-raw` / `--metadata-brief` / `--metadata-all`
control the level of detail.

## Removed compared with upstream diffusion.cpp

- ggml as an external dependency — only the needed kernel tree is kept, and
  it is gk's
- convert / quantize modes and imatrix collection — use the quantizer
- PyTorch checkpoint loading (`.ckpt` / `.pt` / `.pth` pickle + zip)
- WebP image and WebM video output
