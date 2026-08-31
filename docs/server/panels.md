# Server panel — the tabs

Six tabs across the top: **Server**, **Model**, **Settings**, **Presets**,
**Hardware**, **Logs**. Everything you set is written to `localStorage` under
`gguf-server-web-v1` as you type, so a reload keeps your setup.

---

## Server

The control tab. What it shows depends on state.

| Element | What it is |
| ------- | ---------- |
| Big status card | `Stopped` / `Starting…` / `Running` / `Error`, with uptime |
| Start / Stop button | Spawns or terminates the engine process |
| Active Configuration | Model file, alias, context, GPU layers, host:port as launched |
| OpenAI-Compatible Endpoints | The endpoint table below, with the live base URL |
| Command | The exact argument list, and an **Edit manually** toggle |

### Endpoints listed

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/rerank` | Rerank documents against a query |
| `POST` | `/v1/messages` | Anthropic-compatible messages |
| `GET` | `/props` | Model properties and server settings |
| `GET` | `/slots` | State of the inference slots |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

The engine serves more than this — see [engine.md](engine.md).

### Edit manually

Ticking **Edit manually** replaces the whole structured configuration with a
free-text command line. A leading binary name is stripped (`gguf-server`,
`./gguf-server`, `gguf-server.exe`, and the historical `llama-server` /
`server` spellings), because the bundled engine is always the one executed.
Host and port are read back out of `--host` / `--port` so the GUI still knows
where the server will be.

This is the escape hatch for any flag the GUI doesn't expose — router mode,
speculative decoding, LoRA adapters, SSL.

---

## Model

| Control | Flag | Notes |
| ------- | ---- | ----- |
| GGUF Model File | `--model` | **Required.** Picked through the browse dialog |
| Multimodal projector | `--mmproj` | Optional; see [multimodal.md](multimodal.md) |
| Chat Template mode | — | `auto` / `file` / `custom` |
| — template file | `--chat-template-file` | `.json`, `.jinja`, `.jinja2`, `.txt` |
| — template text | `--chat-template` | A built-in name or an inline Jinja template |

`auto` uses the template stored in the model's own metadata, which is right
almost always. Override it when a model ships a broken template or when you
want a different tool-call dialect.

The file browser lists directories plus files matching the tab's filter
(`.gguf` for models, template suffixes for templates). Hidden files are
skipped. It starts at your home directory and remembers where you were.

---

## Settings

### Network

| Control | Flag | Default |
| ------- | ---- | ------- |
| Host | `--host` | `127.0.0.1` (also `0.0.0.0`, `localhost`) |
| Port | `--port` | `8888` |
| API Key | `--api-key` | empty (no auth) |
| Model Alias | `--alias` | model filename stem |

Binding `0.0.0.0` exposes inference to your network. Set an API key if you do.

### Performance

| Control | Flag | Default | Range |
| ------- | ---- | ------- | ----- |
| Context Length | `--ctx-size` | 65536 | 512 – 1048576 |
| GPU Layers | `--n-gpu-layers` | 999 | 0 – 999 |
| Main GPU | `--main-gpu` | 0 | 0 – 31, only sent when GPU layers > 0 |
| Tensor Split | `--tensor-split` | empty | e.g. `24,8`; only with GPU layers > 0 |
| CPU Threads | `--threads` | 0 (auto) | omitted when 0 |
| Parallel Slots | `--parallel` | 1 | omitted when 1 |
| Batch Size | `--batch-size` | 2048 | 1 – 1048576 |
| Micro-batch Size | `--ubatch-size` | 512 | 1 – 1048576 |

`999` GPU layers means "offload everything that fits". `0` means CPU-only,
and also sets `CUDA_VISIBLE_DEVICES=-1` for the child so the choice sticks.

The **Recommended** button on this card derives values from the detected
machine:

| | CUDA present | CPU only |
| --- | --- | --- |
| Context length | 8192 | 4096 |
| GPU layers | 999 | 0 |
| Batch / micro-batch | 2048 / 512 (256 if VRAM < 12 GiB) | 512 / 256 |
| Flash attention | on | off |
| Main GPU | the one with most free VRAM | 0 |
| Tensor split | free-VRAM weights, if ≥ 2 usable GPUs | empty |

### Advanced

| Control | Flag | Default |
| ------- | ---- | ------- |
| Flash Attention | `--flash-attn on\|off` | on |
| KV cache type (K) | `--cache-type-k` | `f16` |
| KV cache type (V) | `--cache-type-v` | `f16` |
| Continuous Batching | `--cont-batching` | on |
| mlock | `--mlock` | off |
| No mmap | `--no-mmap` | off |
| Verbose logging | `--verbose` | off |

KV cache types offered: `f32`, `f16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`,
`bf16`, `q6_k`, `q5_k`, `q4_k`, `q3_k`, `q2_k`, `iq4_nl`, `iq4_xs`, `iq3_s`,
`iq3_xxs`, `iq2_s`, `iq2_xs`, `iq2_xxs`, `iq1_s`, `tq2_0`, `tq1_0`, `mxfp4`.
A value of `f16` is not sent (it is the engine default). Quantized KV usually
wants flash attention on.

`mlock` pins weights in RAM so the OS cannot swap them out — helpful on a
machine under memory pressure, harmful if the model does not fit.
`--no-mmap` reads the whole file up front instead of mapping it, which is
slower to start but avoids page-fault stalls on network filesystems.

---

## Presets

Save the current configuration under a name, and reload or delete it later.
Presets are stored in `localStorage` (`gguf-server-web-presets`), not on
disk, so they are per-browser-profile. A preset captures the whole config
object, including the custom command if one is set.

---

## Hardware

A view of what the machine has. **Refresh** takes one snapshot; **Live**
polls continuously while the tab is open.

| Card | Contents |
| ---- | -------- |
| Hardware Monitor | Whether CUDA was detected, snapshot vs live mode, time of the last sample |
| CPU | Model name, architecture, physical/logical cores, load |
| Total Memory Capacity | RAM total/available, VRAM total/used |
| GPU Devices | Per GPU: name, vendor, bus id, VRAM, utilisation, temperature |

Detection is best-effort and uses only what is already on the machine:

- **NVIDIA** — `nvidia-smi --query-gpu=…`, which gives utilisation and
  temperature as well as memory.
- **Windows** — PowerShell / CIM (`Win32_Processor`, `Win32_VideoController`,
  `Win32_OperatingSystem`) when there is no `nvidia-smi`.
- **macOS** — `sysctl` for CPU brand, core counts and `hw.memsize`.
- **Linux** — `/proc/cpuinfo`, `/proc/meminfo`, `getloadavg`.

A field that could not be read is reported as "not reported" rather than
zero.

> Note: this is the *system's* view of your GPUs. It is **not** the ordering
> the engine uses. See [../hardware.md](../hardware.md).

---

## Logs

The engine's stdout and stderr, merged, tailed from the log file the panel
redirected them into (default: last 200 000 bytes). A dot appears on the tab
label when new lines arrive while you are elsewhere.

The log file itself is a temp file named `gguf-server-*.log`; its path is
shown in the server status payload, so you can `tail -f` it outside the
browser.

When a start fails, the last eight non-empty lines are also folded into the
error message on the Server tab, along with a hint pointing at GPU layers,
the model path, or memory.
