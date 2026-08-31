# `ggk` command-line reference

```
ggk [--host HOST] [--port PORT] [--no-browser]   launch the unified GUI
ggk server   [...]                               the LLM server panel
ggk diffuser [...]                               the diffusion panel
ggk editor   [...]                               the GGUF editor panel
```

`python -m ggk` is identical to `ggk` in every form below.

Each panel subcommand accepts exactly what its standalone package accepted,
so scripts written against `gguf-server`, `gguf-diffusion` or `gguf-editor`
keep working with a `ggk ` prefix.

## `ggk` — unified GUI

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--host HOST` | `127.0.0.1` | Address the GUI binds to |
| `--port PORT` | `8640` | Port the GUI listens on; `0` picks a free one |
| `--no-browser` | off | Don't open a browser window |
| `--version` | | Print the version and exit |

Serves the shell page at `/` and mounts the three panels at `/server/`,
`/diffuser/` and `/editor/`. Ctrl-C or `SIGTERM` shuts it down and stops any
engine child that is still running.

## `ggk server`

```
ggk server [serve] [--host HOST] [--port PORT] [--no-browser]
ggk server engine -- <engine args…>
```

`serve` is the default, so `ggk server --port 9000` works without naming it.

### `ggk server serve`

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--host HOST` | `127.0.0.1` | Address the **GUI** binds to |
| `--port PORT` | `8642` | Port the **GUI** listens on; `0` picks a free one |
| `--no-browser` | off | Don't open a browser window |

The LLM server itself is a separate process on its own port — `8888` by
default, configured in the GUI's **Settings → Network** tab.

### `ggk server engine`

Runs the bundled `gguf-server` binary with the arguments after `--` passed
verbatim, and forwards its exit code. Ctrl-C terminates the child and returns
`130`.

```bash
ggk server engine -- --model model.gguf --port 8888 --ctx-size 8192
ggk server engine -- --help
```

Full flag list: [server/engine.md](server/engine.md).

## `ggk diffuser`

```
ggk diffuser [serve] [--host HOST] [--port PORT] [--no-browser]
ggk diffuser engine -- <engine args…>
```

### `ggk diffuser serve`

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--host HOST` | `127.0.0.1` | Address the GUI binds to |
| `--port PORT` | `8643` | Port the GUI listens on; `0` picks a free one |
| `--no-browser` | off | Don't open a browser window |

### `ggk diffuser engine`

Runs the bundled `diffusion` binary with everything after `--` passed
verbatim.

```bash
ggk diffuser engine -- --diffusion-model flux.gguf --vae ae.gguf \
    --t5xxl t5xxl.gguf --clip_l clip_l.gguf \
    -p "a red cube on a white table" --steps 20 -o out.png

ggk diffuser engine -- --list-devices     # JSON device list --backend accepts
ggk diffuser engine -- --help
```

Full flag list: [diffuser/engine.md](diffuser/engine.md).

## `ggk editor`

```
ggk editor [serve] [MODEL.gguf] [--host HOST] [--port PORT] [--no-browser]
ggk editor quantize -m IN -o OUT [--type T] [--tensor-type-rules R] [--device D] [-t N]
ggk editor devices
```

### `ggk editor serve`

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `MODEL` | — | GGUF file to open on start |
| `--host HOST` | `127.0.0.1` | Address the GUI binds to |
| `--port PORT` | `8644` | Port the GUI listens on; `0` picks a free one |
| `--no-browser` | off | Don't open a browser window |

A bare `ggk editor model.gguf` is `serve` with an initial file.

### `ggk editor quantize`

Runs the quantizer library directly, with no GUI.

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `-m, --model PATH` | required | Input model (`.gguf` or `.safetensors`) |
| `-o, --output PATH` | required | Output GGUF file |
| `--type TYPE` | — | Default weight type for every eligible tensor (e.g. `q4_k`) |
| `--tensor-type-rules RULES` | — | Comma-separated `<regex>=<type>` overrides |
| `--device DEVICE` | `auto` | `cpu`, `auto`, or a name from `ggk editor devices` |
| `-t, --threads N` | `0` | Worker threads; `0` auto-detects |

At least one of `--type` / `--tensor-type-rules` is required. Progress and
warnings go to stderr.

Exit codes: `0` success, `1` error, `2` cancelled.

```bash
# whole model to q4_k
ggk editor quantize -m in.gguf -o out-q4_k.gguf --type q4_k

# q8_0 everywhere, but keep the output head and embeddings at f16
ggk editor quantize -m in.gguf -o out.gguf --type q8_0 \
    --tensor-type-rules '^output\.=f16,^token_embd\.=f16'
```

Supported type names: [editor/quantizer.md](editor/quantizer.md).

### `ggk editor devices`

Lists the devices the quantizer library was compiled with. Index 0 is always
`cpu`.

```
available devices:
  cpu              CPU (built-in quantization kernels)
  cuda0            NVIDIA GeForce RTX 4050 Laptop GPU
```

## Ports at a glance

| Process | Default port |
| ------- | ------------ |
| Unified GUI (`ggk`) | 8640 |
| Server panel GUI (`ggk server`) | 8642 |
| Diffuser panel GUI (`ggk diffuser`) | 8643 |
| Editor panel GUI (`ggk editor`) | 8644 |
| `gguf-server` LLM engine | 8888 |

Pass `--port 0` to any GUI to have the OS pick a free port; the chosen one is
printed on startup.

## Environment variables

Read at install/build time:

| Variable | Meaning |
| -------- | ------- |
| `GGK_CUDA`, `GGK_HIP`, `GGK_VULKAN`, `GGK_METAL` | Enable a gk backend |
| `GGK_OPENSSL` | Link the server engine against OpenSSL (default off) |
| `GGK_SUBPROCESS` | Build router mode support (default on) |
| `GGK_MINGW_STATIC` | Static MinGW runtime on Windows (default on) |
| `GGK_ENGINE_DIR` | Use an engine source tree other than `vendor/engine` |
| `GGUF_SERVER_UI_DIR` | Static assets to embed into the server engine |

Read at run time (by the engines, not by `ggk` itself):

| Variable | Meaning |
| -------- | ------- |
| `GK_QUIET` | Suppress the device banner on stderr |
| `LLAMA_ARG_*` | Every `gguf-server` flag has one; e.g. `LLAMA_ARG_PORT` |
| `LLAMA_API_KEY` | API key for the LLM server |
| `LLAMA_CACHE` | Model download cache directory |
| `SD_TOKENIZER_PACK` | Default `--tokenizer-pack` directory for diffusion |

More gk debug variables are listed in [engine/gk.md](engine/gk.md).
