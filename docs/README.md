# ggk documentation

`ggk` is one Python package for working with GGUF models locally. It ships
three panels on one GUI — an OpenAI-compatible **LLM server**, a **diffusion**
image/video/audio generator, and a **GGUF metadata/tensor editor** with a
built-in quantizer — all driven by one C/C++ engine compiled in a single build
on top of **gk**, an independent tensor library. There is no ggml in the tree.

```
pip install ggk
ggk                 # unified GUI: Server / Diffuser / Editor
```

## Start here

| Document | What it covers |
| -------- | -------------- |
| [install.md](install.md) | `pip install`, GPU switches, wheel builds, platform notes |
| [quickstart.md](quickstart.md) | First run of each panel, end to end |
| [cli.md](cli.md) | Every `ggk` subcommand and flag |
| [gui.md](gui.md) | The unified GUI shell, ports, panel mounting |
| [build.md](build.md) | Building the engine from source with CMake |
| [hardware.md](hardware.md) | Backends, device selection, multi-GPU, memory |
| [troubleshooting.md](troubleshooting.md) | Failures and what they mean |
| [faq.md](faq.md) | Short answers to common questions |
| [development.md](development.md) | Repo layout, how the pieces fit, how to hack on it |

## The three panels

### Server — [docs/server](server/README.md)

An OpenAI-compatible HTTP server for GGUF language models, supervised from a
web GUI.

| Document | What it covers |
| -------- | -------------- |
| [server/README.md](server/README.md) | Overview and where things live |
| [server/panels.md](server/panels.md) | The six tabs: Server, Model, Settings, Presets, Hardware, Logs |
| [server/engine.md](server/engine.md) | The `gguf-server` engine: CLI flags, endpoints, env vars |
| [server/models.md](server/models.md) | Supported LLM architectures and chat templates |
| [server/multimodal.md](server/multimodal.md) | Vision and audio inputs, `mmproj` projectors |
| [server/api.md](server/api.md) | The panel's own local JSON API (`/api/*`) |
| [server/examples.md](server/examples.md) | Worked examples: chat, embeddings, rerank, router mode |

### Diffuser — [docs/diffuser](diffuser/README.md)

Image, video and audio generation from GGUF/safetensors diffusion models.

| Document | What it covers |
| -------- | -------------- |
| [diffuser/README.md](diffuser/README.md) | Overview and where things live |
| [diffuser/panels.md](diffuser/panels.md) | The five tabs: Compose, Workflows, Output, Hardware, Logs |
| [diffuser/engine.md](diffuser/engine.md) | The `diffusion` CLI: full flag reference |
| [diffuser/models.md](diffuser/models.md) | Supported model families and what files each needs |
| [diffuser/api.md](diffuser/api.md) | The panel's own local JSON API (`/api/*`) |
| [diffuser/examples.md](diffuser/examples.md) | Worked examples: txt2img, img2img, inpaint, video, upscale |

### Editor — [docs/editor](editor/README.md)

A GGUF metadata/tensor editor with a built-in quantizer.

| Document | What it covers |
| -------- | -------------- |
| [editor/README.md](editor/README.md) | Overview and where things live |
| [editor/panels.md](editor/panels.md) | The editing surface: tables, drawers, modals |
| [editor/quantizer.md](editor/quantizer.md) | Quantization types, rules, devices, the C API |
| [editor/gguf-format.md](editor/gguf-format.md) | GGUF layout, value types, tensor type traits |
| [editor/api.md](editor/api.md) | The panel's own local JSON API (`/api/*`) |
| [editor/examples.md](editor/examples.md) | Worked examples: retitle, rename, merge, quantize |

## The engine

| Document | What it covers |
| -------- | -------------- |
| [engine/README.md](engine/README.md) | Architecture: gk, the ggml compat layer, the three runtimes |
| [engine/gk.md](engine/gk.md) | The compute library: backends, scheduling, memory, env vars |
| [engine/quantizer.md](engine/quantizer.md) | The `qz_*` block codec, compiled into two places |

Source-tree references that ship with the engine itself:

- `vendor/engine/README.md` — engine build options
- `vendor/engine/app/README.md` — the complete HTTP API reference
- `vendor/engine/gk/README.md` — gk internals
- `vendor/engine/diffusion/README.md` — diffusion runtime notes
- `vendor/engine/mtmd/README.md` — multimodal projectors

## Conventions used in these docs

- Shell examples assume `ggk` is on `PATH`. `python -m ggk` is equivalent.
- Paths are written with `/`; on Windows use `\` (the GUI handles both).
- "Panel" means one of the three GUIs. "Engine" means the compiled C/C++
  artifact a panel drives: `gguf-server`, `diffusion`, `libgguf_quantizer`.
