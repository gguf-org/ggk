# Diffuser panel

Image, video and audio generation from GGUF/safetensors diffusion models,
with a web GUI over the bundled `diffusion` engine.

```bash
ggk                # Diffuser tab of the unified GUI
ggk diffuser       # standalone, http://127.0.0.1:8643/
```

## In this section

| Document | What it covers |
| -------- | -------------- |
| [panels.md](panels.md) | The five tabs and every control on them |
| [engine.md](engine.md) | The `diffusion` CLI: complete flag reference |
| [models.md](models.md) | Supported families and the files each one needs |
| [api.md](api.md) | The panel's own local JSON API |
| [examples.md](examples.md) | Worked examples |

## How it works

```
browser  ──HTTP──▶  ggk diffuser panel (Python, stdlib only)
                        │  spawns one process per generation
                        ▼
                    diffusion  (C/C++, bundled)
                        │  writes  out.png / out_0000.png …
                        ▼
                    gk kernels (CPU / CUDA / HIP / Metal / Vulkan)
```

Each **Generate** click spawns one engine process. A reader thread consumes
its merged stdout/stderr unbuffered, splits on both `\n` and `\r` so the
progress bars arrive live, parses `| 5/20 -` into a step counter, and
collects the produced files when the process exits. Several generations can
be in flight at once — each is a separate job with its own id.

Unlike the server panel, nothing is long-lived: there is no daemon to start
or stop, and closing the GUI does not interrupt anything already running.

**Nothing is uploaded.** Models and input images are addressed by filesystem
path through the `/api/browse` listing.

## Files

| Path | What |
| ---- | ---- |
| `src/ggk/diffuser/__main__.py` | The `ggk diffuser` CLI |
| `src/ggk/diffuser/server.py` | HTTP backend: static files + `/api/*` |
| `src/ggk/diffuser/engine.py` | Argument building, job supervision, device probe |
| `src/ggk/diffuser/static/` | The web GUI |
| `src/ggk/diffuser/bin/diffusion` | The compiled engine (installed by the build) |

## Run modes

The engine has four (`-M, --mode`):

| Mode | What it does |
| ---- | ------------ |
| `img_gen` | Image generation — the default, and what the GUI drives |
| `vid_gen` | Video generation; writes an image sequence or AVI |
| `upscale` | Run an ESRGAN-family upscaler over an image |
| `metadata` | Print the generation metadata embedded in an image |

The GUI covers `img_gen`; the other three are reachable through the
**Command → Edit manually** box or `ggk diffuser engine`.

## Output

Images are written to the configured output directory (default
`~/Downloads`, falling back to your home directory) as
`gguf-YYYYmmdd-HHMMSS.png`. A batch produces `gguf-….png` plus
`gguf-…_0000.png`, `_0001.png` … — the panel globs for both patterns when the
process exits.

Output formats the engine can write: **PNG**, **JPG**, **BMP**, **TGA**.
Video modes write image sequences or AVI. WebP and WebM are not supported.

Generation parameters are embedded in the image metadata unless
`--disable-image-metadata` is set; `--mode metadata` reads them back.
