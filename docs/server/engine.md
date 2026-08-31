# The `gguf-server` engine

The LLM server is a single C/C++ executable built from `vendor/engine/` and
installed at `src/ggk/server/bin/gguf-server`. It is the same program whether
you drive it from the GUI or the command line.

```bash
ggk server engine -- --model model.gguf
# → http://127.0.0.1:8888
```

Two defaults differ from upstream `llama-server`: the binary is called
`gguf-server`, and the listen port is **8888**. Everything else — flag names,
`LLAMA_ARG_*` environment variables, response shapes — is unchanged.

> The complete, auto-generated flag table and the full HTTP API reference
> ship with the engine at **`vendor/engine/app/README.md`**. This page is the
> orientation and the subset you will actually reach for.

## Common flags

| Flag | Meaning |
| ---- | ------- |
| `-m, --model FNAME` | GGUF model file |
| `-c, --ctx-size N` | Prompt context size; `0` = read from the model |
| `-ngl, --n-gpu-layers N` | Layers to offload to the GPU |
| `-mg, --main-gpu N` | Primary GPU when splitting |
| `-ts, --tensor-split SPLIT` | Proportions across GPUs, e.g. `24,8` |
| `-t, --threads N` | CPU threads for generation; `-1` = auto |
| `-b, --batch-size N` | Logical batch size (default 2048) |
| `-ub, --ubatch-size N` | Physical micro-batch size (default 512) |
| `-fa, --flash-attn on\|off\|auto` | Flash attention |
| `-ctk, --cache-type-k TYPE` | KV cache K type |
| `-ctv, --cache-type-v TYPE` | KV cache V type |
| `--mlock` | Lock the model in RAM |
| `--no-mmap` | Read instead of memory-mapping the model |
| `-v, --verbose` | Verbose logging |

## Server flags

| Flag | Meaning |
| ---- | ------- |
| `--host HOST` | Listen address; a `.sock` suffix binds a UNIX socket |
| `--port PORT` | Listen port (default **8888**) |
| `-a, --alias STRING` | Model name(s) reported by the API, comma-separated |
| `--api-key KEY` | API key(s), comma-separated |
| `--api-key-file FNAME` | API keys from a file, one per line |
| `-np, --parallel N` | Number of server slots; `-1` = auto |
| `-cb, --cont-batching` | Continuous (dynamic) batching; enabled by default |
| `--api-prefix PREFIX` | Serve from a path prefix |
| `--cors-origins ORIGINS` | Allowed CORS origins (default `*`) |
| `--metrics` | Enable `GET /metrics` |
| `--slots, --no-slots` | Expose `GET /slots` (enabled by default) |
| `--props` | Allow changing global properties via `POST /props` |
| `--embedding, --embeddings` | Restrict to embeddings (for embedding models) |
| `--rerank, --reranking` | Enable the reranking endpoint |
| `--ssl-key-file`, `--ssl-cert-file` | PEM key/cert for HTTPS |
| `-to, --timeout N` | Read/write timeout, seconds (default 3600) |
| `--sleep-idle-seconds N` | Sleep after N idle seconds; `-1` disables |

## Chat and reasoning

| Flag | Meaning |
| ---- | ------- |
| `--jinja, --no-jinja` | Jinja chat templating (enabled by default) |
| `--chat-template NAME_OR_JINJA` | Built-in template name, or an inline template |
| `--chat-template-file PATH` | Template from a file |
| `--chat-template-kwargs JSON` | Extra params for the template parser |
| `-rea, --reasoning on\|off\|auto` | Reasoning/thinking mode |
| `--reasoning-format FORMAT` | `none`, `deepseek`, `deepseek-legacy`, `auto` |
| `--reasoning-budget N` | Thinking token budget; `-1` unrestricted, `0` immediate end |
| `--reasoning-preserve` | Keep reasoning traces in full history |
| `--skip-chat-parsing` | Force a pure content parser |
| `--prefill-assistant` | Prefill when the last message is from the assistant |

Built-in template names are listed in [models.md](models.md).

## Multimodal

| Flag | Meaning |
| ---- | ------- |
| `-mm, --mmproj FILE` | Multimodal projector file |
| `--mmproj-offload, --no-mmproj-offload` | GPU-offload the projector (default on) |
| `--image-min-tokens N`, `--image-max-tokens N` | Token budget per image, for dynamic-resolution vision models |
| `--mtmd-batch-max-tokens N` | Image tokens per encode batch (default 1024) |
| `--media-path PATH` | Directory for `file://` media references |

See [multimodal.md](multimodal.md).

## Caching and context

| Flag | Meaning |
| ---- | ------- |
| `--cache-prompt, --no-cache-prompt` | Prompt caching (enabled by default) |
| `--cache-reuse N` | Minimum chunk to reuse via KV shifting |
| `-cram, --cache-ram N` | Max cache size in MiB (default 8192, `-1` unlimited, `0` off) |
| `-ctxcp, --ctx-checkpoints N` | Context checkpoints per slot (default 32) |
| `-kvu, --kv-unified` | One unified KV buffer shared across sequences |
| `--context-shift` | Context shift for infinite generation (off by default) |
| `--slot-save-path PATH` | Where `POST /slots/{id}?action=save` writes |
| `-sps, --slot-prompt-similarity S` | Prompt-match threshold for slot reuse (default 0.10) |

## Router mode — serving several models

Built when `GGK_SUBPROCESS=ON` (the default). The parent process spawns a
child server per model and routes by the request's `model` field.

| Flag | Meaning |
| ---- | ------- |
| `--models-dir PATH` | Directory of GGUF files to serve |
| `--models-preset PATH` | INI file of model presets |
| `--models-max N` | Max models loaded at once (default 4, `0` unlimited) |
| `--models-autoload` | Load on demand (enabled by default) |

```bash
ggk server engine -- --models-dir ~/models --models-autoload
```

Router mode adds `GET /models`, `POST /models/load`, `POST /models/unload`,
`GET /models/sse`, `POST /models` (download) and `DELETE /models`.

## Built-in tools

`--tools TOOL1,TOOL2,…` (or `-ag, --agent` for everything) enables
server-side tools for agent workflows: `read_file`, `file_glob_search`,
`grep_search`, `exec_shell_command`, `write_file`, `edit_file`,
`get_datetime`.

These execute on the server machine. Enabling them limits `--cors-origins`
to localhost by default. Do not enable them in an untrusted environment.

## HTTP endpoints

### Native

| Method | Path |
| ------ | ---- |
| `GET` | `/health` |
| `POST` | `/completion` |
| `POST` | `/tokenize`, `/detokenize`, `/apply-template` |
| `POST` | `/embedding`, `/embeddings`, `/reranking` |
| `POST` | `/infill` |
| `GET`/`POST` | `/props` |
| `GET` | `/slots` |
| `POST` | `/slots/{id}?action=save\|restore\|erase` |
| `GET` | `/metrics` |
| `GET`/`POST` | `/lora-adapters` |

### OpenAI-compatible

| Method | Path |
| ------ | ---- |
| `GET` | `/v1/models` |
| `POST` | `/v1/completions` |
| `POST` | `/v1/chat/completions` |
| `POST` | `/v1/chat/completions/control` |
| `POST` | `/v1/responses` |
| `POST` | `/v1/embeddings` |
| `POST` | `/v1/rerank`, `/v1/reranking` |
| `POST` | `/v1/audio/transcriptions` |
| `POST` | `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens` |
| `GET` | `/v1/health` |

### Anthropic-compatible

| Method | Path |
| ------ | ---- |
| `POST` | `/v1/messages` |
| `POST` | `/v1/messages/count_tokens` |

Request and response schemas for every one of these are documented in
`vendor/engine/app/README.md`.

## Environment variables

Every flag has an `LLAMA_ARG_*` twin — the names are unchanged from upstream,
so existing deployment scripts work:

```bash
LLAMA_ARG_MODEL=model.gguf LLAMA_ARG_PORT=9000 LLAMA_ARG_CTX_SIZE=8192 \
    ggk server engine
```

| Variable | Flag |
| -------- | ---- |
| `LLAMA_ARG_MODEL` | `--model` |
| `LLAMA_ARG_PORT` | `--port` |
| `LLAMA_ARG_HOST` | `--host` |
| `LLAMA_ARG_CTX_SIZE` | `--ctx-size` |
| `LLAMA_ARG_N_GPU_LAYERS` | `--n-gpu-layers` |
| `LLAMA_ARG_BATCH` / `LLAMA_ARG_UBATCH` | `--batch-size` / `--ubatch-size` |
| `LLAMA_ARG_FLASH_ATTN` | `--flash-attn` |
| `LLAMA_ARG_MMPROJ` | `--mmproj` |
| `LLAMA_ARG_JINJA` | `--jinja` |
| `LLAMA_ARG_CHAT_TEMPLATE` | `--chat-template` |
| `LLAMA_API_KEY` | `--api-key` |
| `LLAMA_CACHE` | Model download cache directory |
| `GK_QUIET` | Suppress the gk device banner |

`gguf-server --help` prints the authoritative list, each entry annotated with
its environment variable.

## The web UI

The engine can embed a static web UI, but this tree ships no asset bundle, so
a plain build is **API-only** and `GET /` returns 404. That is deliberate:
`ggk` serves its own GUI.

To get one anyway:

```bash
# embed at build time
CMAKE_ARGS="-DGGUF_SERVER_UI_DIR=/path/to/dist" pip install ggk
# or serve from disk at run time
ggk server engine -- --model model.gguf --path /path/to/dist
```
