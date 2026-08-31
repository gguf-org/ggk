# Supported models — diffuser panel

The engine loads **GGUF** and **safetensors** weights. It detects the family
from the tensor names in the file, so you do not declare it — but you do have
to supply the right *companion* files, because most modern families ship the
diffusion transformer, the VAE and the text encoder(s) separately.

PyTorch checkpoints (`.ckpt`, `.pt`, `.pth`) are **not** supported. Convert
or download a GGUF/safetensors build.

## Detected families

| Family | Variants the engine distinguishes |
| ------ | --------------------------------- |
| **Stable Diffusion 1.x** | base, inpaint, pix2pix, tiny-UNet, SDXS-512-DS |
| **Stable Diffusion 2.x** | base, inpaint, tiny-UNet, SDXS-0.9 |
| **SDXL** | base, inpaint, pix2pix, Vega, SSD-1B |
| **SVD** | Stable Video Diffusion |
| **SD3 / SD3.5** | `VERSION_SD3` |
| **Flux** | Flux, Flux Fill, Flux Controls, Flex.2, Chroma, Chroma Radiance, Ovis Image |
| **Flux 2** | Flux2, Flux2 Klein |
| **Qwen-Image** | Qwen-Image, Qwen-Image-Layered |
| **Wan** | Wan 2.x, Wan 2.2 I2V, Wan 2.2 TI2V |
| **HunyuanVideo** | `VERSION_HUNYUAN_VIDEO` |
| **LTX-AV** | `VERSION_LTXAV` |
| **Lingbot Video** | `VERSION_LINGBOT_VIDEO` |
| **HiDream** | HiDream-O1 |
| **Z-Image** | `VERSION_Z_IMAGE` |
| **PixArt** | `VERSION_PIXART` |
| **Lumina 2** | `VERSION_LUMINA2` |
| **Anima** | `VERSION_ANIMA` |
| **Boogu Image** | `VERSION_BOOGU_IMAGE` |
| **ERNIE Image** | `VERSION_ERNIE_IMAGE` |
| **Lens** | `VERSION_LENS` |
| **MiniT2I** | `VERSION_MINIT2I` |
| **LongCat** | `VERSION_LONGCAT` |
| **PiD** | `VERSION_PID` |
| **Ideogram 4** | `VERSION_IDEOGRAM4` |
| **Sefi Image** | `VERSION_SEFI_IMAGE` |
| **Krea 2** | `VERSION_KREA2` |
| **Mage Flow** | `VERSION_MAGE_FLOW` |
| **MiniMax H3** | `VERSION_MINIMAX_H3` |
| **ACE-Step** | audio generation, with lyrics |
| **MiniMax Music 3** | audio generation |
| **ESRGAN** | upscalers (`--mode upscale`, `--upscale-model`, hires fix) |

Derivatives built on these backbones — Turbo, LCM, Lightning, Hyper,
NitroFusion, distilled and Schnell variants — load as their base family. What
changes is the recipe: fewer steps, CFG at 1.0, sometimes a specific
scheduler or `--timestep-shift`.

## What each family needs

The text encoder is the part people most often get wrong. The engine picks
its conditioner from the detected family:

| Family | Text encoder flags | Notes |
| ------ | ------------------ | ----- |
| SD 1.x / 2.x | built into the checkpoint, or `--clip_l` | CLIP-L (SD2 uses OpenCLIP) |
| SDXL | `--clip_l` + `--clip_g` | Dual CLIP |
| SD3 / SD3.5 | `--clip_l` + `--clip_g` + `--t5xxl` | T5 is optional but strongly recommended |
| Flux | `--clip_l` + `--t5xxl` | |
| Chroma / Chroma Radiance | `--t5xxl` | T5 only — detected by a `distilled_guidance_layer` tensor |
| Flux 2 | `--llm` = Mistral Small 3.2 | |
| Flux 2 Klein | `--llm` = Qwen3 | |
| Ovis Image | `--llm` = Qwen3 | |
| Qwen-Image / Qwen-Image-Layered | `--llm` = Qwen2.5-VL (+ `--llm_vision` for edit) | |
| Z-Image | `--llm` = Qwen3 | |
| Wan 2.x | `--t5xxl` = UMT5 | Mask always on |
| HunyuanVideo | `--llm` (+ optional ByT5) | |
| LTX-AV | `--llm` + `--embeddings-connectors` (+ `--audio-vae`) | |
| PixArt | `--t5xxl`, or `--llm` Qwen3-0.6B with `--llm-adapter` | |
| Lumina 2 / PiD | `--llm` = Gemma 2 2B | |
| ERNIE Image | `--llm` = Ministral 3B | |
| Lens | `--llm` = GPT-OSS 20B | |
| Ideogram 4, Lingbot Video, Boogu, Sefi, Krea 2, Mage Flow, MiniMax H3 | `--llm` = Qwen3-VL | |
| ACE-Step | built-in lyric tokenizer | `--lyrics`, `--audio-duration` |

A **VAE** (`--vae`) is needed whenever it is not inside the main file — which
is most of the time for standalone `--diffusion-model` weights. Chroma
Radiance is pixel-space and needs none. `--vae-format` (`auto`, `flux`,
`sd3`, `flux2`) overrides latent-format detection if the file is ambiguous.

`--taesd` swaps in a Tiny AutoEncoder for a much faster, much lower-quality
decode — good for iterating on prompts, not for final output.

### `--model` vs `--diffusion-model`

| Use | When |
| --- | ---- |
| `-m, --model` | One file contains everything (a classic SD1.x/SDXL checkpoint) |
| `--diffusion-model` | The file is only the transformer/UNet — the usual case for Flux, SD3, Qwen-Image, Wan, and every GGUF quantized release |

The GUI's Model card has this as a toggle.

## Video models

| Family | Notes |
| ------ | ----- |
| Wan 2.x | T2V and I2V; Wan 2.2 adds a two-expert MoE — pass `--high-noise-diffusion-model` and tune `--moe-boundary` |
| HunyuanVideo | |
| LTX-AV | Needs `--embeddings-connectors`; `--audio-vae` for the audio track |
| Lingbot Video | |
| SVD | Image-to-video |

Video runs want `--mode vid_gen`, `--video-frames`, `--fps`, and usually
`--vae-tiling` plus `--temporal-tiling` to keep memory in range. First/last
frame interpolation uses `-i` and `--end-img`.

## Audio models

| Family | Flags |
| ------ | ----- |
| ACE-Step | `--lyrics`, `--audio-duration` |
| MiniMax Music 3 | `--lyrics`, `--audio-vae` |
| LTX-AV (audio track) | `--audio-vae` |

Leaving `--lyrics` empty produces instrumental output.

## Adapters and extras

| What | Flags |
| ---- | ----- |
| ControlNet | `--control-net`, `--control-image` / `--control-video`, `--control-strength` |
| LoRA | `--lora-model-dir`, referenced as `<lora:name:weight>` in the prompt; `--lora-apply-mode` |
| Textual inversion | `--embd-dir`, referenced by name in the prompt |
| PhotoMaker | `--photo-maker`, `--pm-id-images-dir`, `--pm-id-embed-path`, `--pm-style-strength` |
| PuLID | `--pulid-weights`, `--pulid-id-embedding`, `--pulid-id-weight` |
| Upscalers (ESRGAN) | `--upscale-model`, `--upscale-repeats`, `--upscale-tile-size` |
| Hires fix | `--hires` and the `--hires-*` family |

## On-the-fly quantization

The engine can quantize weights as it loads them, without writing a new file:

```bash
ggk diffuser engine -- --diffusion-model flux-f16.safetensors --type q8_0 …
ggk diffuser engine -- --diffusion-model flux.safetensors \
    --tensor-type-rules '^vae\.=f16,model\.=q4_k' …
```

For a persistent quantized file, use the
[Editor panel](../editor/quantizer.md) instead — quantizing once beats
quantizing on every run.

## Starting-point recipes

Treat these as sane defaults, not gospel; every release retunes something.

| Model | Steps | CFG | Sampler | Schedule |
| ----- | ----- | --- | ------- | -------- |
| SD 1.5 | 20–30 | 7.0 | `euler_a` | `discrete` |
| SDXL | 25–40 | 6.0–8.0 | `dpm++2m` | `karras` |
| SDXL Turbo / LCM | 1–8 | 1.0 | `euler_a` / `lcm` | `discrete` / `lcm` |
| SD3.5 | 28 | 4.5 | `euler` | `discrete` (try `--slg-scale 2.5` on medium) |
| Flux Dev | 20–28 | 1.0 | `euler` | `discrete` (guidance via `--guidance 3.5`) |
| Flux Schnell | 4 | 1.0 | `euler` | `discrete` |
| Qwen-Image | 20–30 | 4.0 | `euler` | `discrete` |
| Wan 2.x video | 20–40 | 5.0 | `euler` | `discrete` (`--flow-shift` matters) |

For distilled models, CFG **must** be 1.0 — a higher value both doubles the
compute and degrades output, because the negative branch was trained away.

## Verifying what a file is

```bash
# what the engine detects, plus every tensor it found
ggk diffuser engine -- --diffusion-model model.gguf -v -p "" --steps 1 -o /tmp/probe.png

# what generated an existing image
ggk diffuser engine -- --mode metadata --image out.png
```

The [Editor panel](../editor/README.md) also opens diffusion GGUF files and
shows their metadata and tensor names directly.
