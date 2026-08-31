# Server panel

An OpenAI-compatible HTTP server for GGUF language models, with a web GUI
that configures, starts, supervises and tails it.

```bash
ggk                # Server tab of the unified GUI
ggk server         # standalone, http://127.0.0.1:8642/
```

## In this section

| Document | What it covers |
| -------- | -------------- |
| [panels.md](panels.md) | The six tabs and every control on them |
| [engine.md](engine.md) | The `gguf-server` engine: flags, endpoints, env vars |
| [models.md](models.md) | Supported architectures, quant types, chat templates |
| [multimodal.md](multimodal.md) | Vision/audio inputs and `mmproj` projectors |
| [api.md](api.md) | The panel's own local JSON API |
| [examples.md](examples.md) | Worked examples |

## How it works

```
browser  ──HTTP──▶  ggk server panel (Python, stdlib only)
                        │  spawns / supervises
                        ▼
                    gguf-server  (C/C++, bundled)  ──▶  :8888/v1/...
                        │
                        ▼
                    gk kernels (CPU / CUDA / HIP / Metal / Vulkan)
```

The Python side never touches a tensor. It builds an argument list, spawns
one engine process with its output redirected to a log file, polls the TCP
port until it accepts connections, and streams the log tail to the GUI. Your
API clients talk to the engine directly.

**One server at a time.** Starting a new one stops the current one first and
waits for the port to be released.

**Nothing is uploaded.** The panel runs on the same machine as the browser,
so models are addressed by filesystem path. The GUI picks paths through a
directory listing API rather than drag & drop — a dropped multi-GB model
would have to be copied into temp storage first.

## Files

| Path | What |
| ---- | ---- |
| `src/ggk/server/__main__.py` | The `ggk server` CLI |
| `src/ggk/server/server.py` | HTTP backend: static files + `/api/*` |
| `src/ggk/server/engine.py` | Argument building, process supervision, hardware probe |
| `src/ggk/server/static/` | The web GUI (`index.html`, `app.js`, `style.css`) |
| `src/ggk/server/bin/gguf-server` | The compiled engine (installed by the build) |

## Lifecycle of a run

1. **Start** — the GUI `POST`s its config to `/api/start`.
2. The backend validates paths, builds the argument list, stops any current
   server, and waits for the port to free up (up to 3 s).
3. `ServerProcess` spawns the binary with `stdout`/`stderr` into a temp log
   file, `stdin` at `DEVNULL`, and `cwd` set to the binary's own directory so
   Windows finds sibling DLLs.
4. A watcher thread polls the port every 500 ms for up to 120 s.
   - port accepts → `running`
   - process exits first → `error`, with the last few log lines and a hint
   - neither → `error` after the timeout, suggesting the model may be too
     large for available memory
5. The GUI polls `/api/server` for status and `/api/log` for the log tail.
6. **Stop** sends `SIGTERM`, waits 5 s, then `SIGKILL`s.

If `--n-gpu-layers 0` is configured, `CUDA_VISIBLE_DEVICES=-1` is set for the
child, so an explicit CPU-only choice really is CPU-only.

## Where the API lives

Once running, everything is on the engine's own host/port — `8888` by
default:

```
http://127.0.0.1:8888/v1/chat/completions
http://127.0.0.1:8888/v1/models
http://127.0.0.1:8888/health
```

The **Server** tab shows the exact base URL, the model id, and the full
command line it launched.
