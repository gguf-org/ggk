# Diffuser panel — the tabs

Five tabs: **Compose**, **Workflows**, **Output**, **Hardware**, **Logs**.
Everything you set is saved to `localStorage` (`gguf-diffusion-web-v1`) as
you type.

---

## Compose

The main tab, top to bottom. Each control shows the engine flag it maps to,
so the GUI doubles as a readable CLI builder.

### Model — *required*

| Control | Flag | Notes |
| ------- | ---- | ----- |
| Mode toggle | `--diffusion-model` / `--model` | `--diffusion-model` for a bare transformer/UNet file; `--model` for an all-in-one checkpoint |
| Model file | (above) | `.gguf` or `.safetensors` |
| VAE | `--vae` | The latent decoder. Most modern families need one |
| Tokenizer pack | `--tokenizer-pack` | A **directory** of replacement tokenizer data, `<family>/{merges.txt,vocab.json,tokenizer.json}` |

The tokenizer pack overrides the engine's built-in tokenizer data. It must be
an existing directory or the start is rejected.

### Text Encoders — *optional*

Add one row per encoder the family needs:

| Flag | Typical use |
| ---- | ----------- |
| `--clip_l` | CLIP-L, SD 1.x/2.x/XL/3.x, Flux |
| `--clip_g` | CLIP-G, SDXL and SD3 |
| `--t5xxl` | T5-XXL, SD3 / Flux / PixArt |
| `--llm` | An LLM text encoder — Qwen2.5-VL for Qwen-Image, Mistral Small 3.2 for Flux2 |
| `--llm_vision` | The LLM's vision tower |
| `--clip_vision` | A CLIP vision encoder |
| `--llm-adapter` | A Qwen3 → T5-XXL bridge, so PixArt can run on `--llm qwen3-0.6b` |

### Additional Models — *optional*

| Flag | What |
| ---- | ---- |
| `--high-noise-diffusion-model` | The high-noise expert of a two-stage MoE model (Wan 2.2) |
| `--uncond-diffusion-model` | Unconditional model for Ideogram4 CFG |
| `--control-net` | A ControlNet |
| `--photo-maker` | PhotoMaker weights |
| `--upscale-model` | An ESRGAN-family upscaler |
| `--taesd` | Tiny AutoEncoder — fast, low-quality decode; good for previews |
| `--embeddings-connectors` | LTXAV embeddings connectors |
| `--audio-vae` | Audio VAE for LTXAV / MiniMax H3 |

### Image Inputs — *optional, collapsed by default*

| Control | Flag | Notes |
| ------- | ---- | ----- |
| Init image | `--init-img` | img2img source |
| Strength | `--strength` | 0–1, how far from the init image (default 0.75) |
| Mask image | `--mask` | Inpainting mask |
| End image | `--end-img` | First-frame/last-frame video interpolation |
| Control image | `--control-image` | ControlNet conditioning |
| Control strength | `--control-strength` | 0–1 (default 0.9) |
| Reference images | `--ref-image` | Repeatable; Flux Kontext and edit models |
| Increase ref index | `--increase-ref-index` | Number references in listed order from 1 |
| Disable ref auto-resize | `--disable-auto-resize-ref-image` | Keep reference images at their native size |

Accepted image types: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`.

### Prompt

| Control | Flag |
| ------- | ---- |
| Positive prompt | `-p` |
| Negative prompt | `-n` |

Negative prompts do nothing on distilled/guidance-free models (Flux Schnell,
Turbo/LCM variants) run at CFG 1.0.

### Generation

| Control | Flag | Default | Range |
| ------- | ---- | ------- | ----- |
| CFG Scale | `--cfg-scale` | 1.0 | 0–20 |
| Steps | `--steps` | 4 | 1–100 |
| Width | `-W` | 512 | 64–2048, clamped |
| Height | `-H` | 512 | 64–2048, clamped |
| Seed | `--seed` | -1 (random) | `-1` omits the flag |
| Batch Count | `--batch-count` | 1 | 1–32; only sent when > 1 |
| Sampling Method | `--sampling-method` | `euler` | see below |
| Schedule | `--scheduler` | `discrete` | see below; `discrete` is not sent |

**Sampling methods:** `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`,
`dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`,
`res_multistep`, `res_2s`, `er_sde`, `euler_cfg_pp`, `euler_a_cfg_pp`,
`euler_ge`

**Schedules:** `discrete`, `karras`, `exponential`, `ays`, `gits`,
`sgm_uniform`, `simple`, `smoothstep`, `kl_optimal`, `lcm`, `bong_tangent`,
`ltx2`, `logit_normal`, `flux2`, `flux`, `beta`

An unknown value in either field is rejected before the engine is spawned,
and both tables are kept in step with the engine's own
`sample_method_to_str[]` / `scheduler_to_str[]`.

### Flags

| Control | Flag | What |
| ------- | ---- | ---- |
| Flash attention | `--diffusion-fa` | Flash attention in the diffusion model |
| Verbose | `-v` | Verbose engine logging |
| Memory-map model | `--mmap` | Map weights instead of reading them |
| Offload weights to CPU | `--offload-to-cpu` | Keep weights in RAM, stream to VRAM on use |
| CLIP on CPU | `--clip-on-cpu` | Text encoders on the CPU |
| VAE on CPU | `--vae-on-cpu` | Decode on the CPU |
| ControlNet on CPU | `--control-net-cpu` | ControlNet on the CPU |

The three "on CPU" flags are the old spelling; the engine now prefers
`--backend te=cpu`, `--backend vae=cpu`, `--backend controlnet=cpu`. The GUI
still emits the short flags because they are unambiguous.

### Tensor Split (multi-GPU)

A collapsed section under **Flags**. Rows of *module → device(s) → split
mode*, which compile to:

```
--backend  diffusion=cuda0&cuda1,vae=cpu
--split-mode diffusion=row
```

| Module | What it covers |
| ------ | -------------- |
| `diffusion` | The main transformer/UNet |
| `te` | Text encoders |
| `clip-vision` | The CLIP vision tower |
| `vae` | The latent decoder |
| `control-net` | ControlNet |
| `photo-maker` | PhotoMaker |
| `upscaler` | The upscale model |

Split modes: `layer` (whole transformer blocks per device, the default) or
`row` (matmul rows split across devices, CUDA only). `--split-mode` is only
emitted for rows that ask for `row`.

Device names come from the engine itself via `/api/devices`. **Use those
names, not `nvidia-smi`'s.** See [../hardware.md](../hardware.md) for why the
two orderings disagree.

### Output

The output directory. Created if missing. Default `~/Downloads`, or your home
directory when that does not exist.

### Command

A live preview of the exact argument list. **Edit manually** replaces the
structured config with free text; a leading `diffusion` / `./diffusion` /
`diffusion.exe` / `diffusion-cli` token is stripped, because the bundled
engine is always the one executed.

This is how you reach anything the GUI does not expose — video mode, hires
fix, LoRA directories, caching, PuLID, `--max-vram`.

### Generate / Cancel

**Generate** posts the config and starts polling the job. Progress comes from
the engine's own progress bar; the step counter and percentage update live.
**Cancel** sends `SIGTERM`, then `SIGKILL` after 5 s, and the job ends as
`cancelled`.

---

## Workflows

Save the current Compose state under a name; reload or delete it later.
Stored in `localStorage` (`gguf-diffusion-web-workflows`), so they are
per-browser-profile. A workflow captures every field, including model paths
and the custom command.

---

## Output

A gallery of generated images, newest first, with the prompt and timestamp.
Clicking one opens it full size. The last 50 entries are remembered across
reloads (`gguf-diffusion-web-images`) as **paths**, not data — deleting a
file on disk leaves a dead entry.

Images are served by `GET /api/image?path=…`, which only serves files whose
suffix is a known image type.

---

## Hardware

CPU count, RAM (Linux `/proc/meminfo`), and NVIDIA GPUs via `nvidia-smi`:
name, VRAM total/used, utilisation, temperature.

This is the *system's* view. The device names `--backend` accepts come from
the engine and are shown in the device dropdowns on the Compose tab.

---

## Logs

The engine's merged output for the current job. Progress-bar redraws are
collapsed: a redraw of the *same* step overwrites the previous line, but once
the step number advances the line is kept, so the polling cursor never gets
stuck showing step 1 forever.

The log is fetched incrementally — the client sends the index it has seen and
gets only what is new.
