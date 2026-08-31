# Server panel — local JSON API

This is the API of the **panel** (the Python control plane), not of the LLM
engine. It exists so the web GUI can configure, start, stop and watch the
engine. For the inference API, see [engine.md](engine.md).

Base URL:

- standalone: `http://127.0.0.1:8642/api/…`
- under the unified GUI: `http://127.0.0.1:8640/server/api/…`

Responses are JSON with `Cache-Control: no-store`. Errors are
`{"error": "message"}` with a 4xx/5xx status.

---

## `GET /api/status`

Static facts about this build. Called once on load.

```json
{
  "version": "0.5.1",
  "engine_available": true,
  "engine_error": null,
  "engine_path": "/…/ggk/server/bin/gguf-server",
  "engine_name": "gguf-server",
  "cache_types": ["f32", "f16", "q8_0", "…"],
  "host_options": ["127.0.0.1", "0.0.0.0", "localhost"],
  "endpoints": [{"method": "GET", "path": "/v1/models", "desc": "List available models"}],
  "defaults": { "host": "127.0.0.1", "port": 8888, "context_length": 65536, "…": "…" },
  "home": "/Users/you",
  "windows": false
}
```

`engine_available: false` means the binary was not found at
`ggk/server/bin/gguf-server` — the engine did not build during install, and
`engine_error` says so.

---

## `GET /api/hardware`

A best-effort hardware snapshot. See
[panels.md § Hardware](panels.md#hardware) for how each field is obtained.

```json
{
  "os": "Darwin 25.6.0",
  "cpu": {
    "name": "Apple M3 Pro", "architecture": "arm64",
    "physical_cores": 12, "logical_cores": 12, "usage_percent": null
  },
  "gpus": [
    {
      "name": "NVIDIA GeForce RTX 4050 Laptop GPU", "vendor": "NVIDIA",
      "device_id": "GPU-…", "bus_id": "00000000:01:00.0",
      "vram_total_mib": 6140, "vram_used_mib": 512,
      "utilization_percent": 3.0, "temperature_c": 41.0, "cuda": true
    }
  ],
  "memory": {
    "total_ram_mib": 36864, "available_ram_mib": 21000,
    "total_vram_mib": 6140, "used_vram_mib": 512
  },
  "cuda_available": true,
  "sampled_at_ms": 1756600000000
}
```

`null` means "not reported on this platform", never zero.

---

## `GET /api/recommended`

The same snapshot plus derived settings:

```json
{ "hardware": { "…": "…" }, "settings": { "context_length": 8192, "gpu_layers": 999, "…": "…" } }
```

The derivation rules are in [panels.md § Settings](panels.md#settings).

---

## `POST /api/browse`

Directory listing for the file picker.

```json
{ "path": "/Users/you/models", "kind": "model" }
```

| Field | Meaning |
| ----- | ------- |
| `path` | Directory to list; omit for the home directory. A file path lists its parent. |
| `kind` | `model` (`.gguf`), `template` (`.json`/`.jinja`/`.jinja2`/`.txt`), `dir` (directories only), `any` |

```json
{
  "path": "/Users/you/models",
  "parent": "/Users/you",
  "entries": [
    {"name": "quants", "path": "/Users/you/models/quants", "is_dir": true},
    {"name": "llama-3.gguf", "path": "/Users/you/models/llama-3.gguf", "is_dir": false, "size": 4661211808}
  ]
}
```

Directories sort first, then files, both case-insensitively. Dotfiles are
skipped. An unreadable directory returns an empty `entries` list rather than
an error.

---

## `POST /api/start`

Start the engine. Stops any current one first.

```json
{
  "config": {
    "model_path": "/Users/you/models/llama-3.gguf",
    "mmproj_path": "",
    "host": "127.0.0.1",
    "port": 8888,
    "context_length": 8192,
    "gpu_layers": 999,
    "main_gpu": 0,
    "tensor_split": "",
    "cpu_threads": 0,
    "n_parallel": 1,
    "batch_size": 2048,
    "ubatch_size": 512,
    "api_key": "",
    "model_alias": "",
    "flash_attn": true,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "cont_batching": true,
    "mlock": false,
    "no_mmap": false,
    "verbose_log": false,
    "chat_template_mode": "auto",
    "chat_template_file": "",
    "chat_template": "",
    "use_custom_command": false,
    "custom_command": ""
  }
}
```

With `use_custom_command: true`, everything except `custom_command` and
`model_alias` is ignored; the command is tokenised with `shlex`, a leading
binary name is stripped, and host/port are read back out of it.

Returns the server state (below). `400` with `{"error": …}` for a missing or
non-existent model file, a missing `mmproj`, an unknown KV cache type, a
missing chat template file, or an empty custom command.

---

## `POST /api/stop`

Terminates the engine (`SIGTERM`, then `SIGKILL` after 5 s) and returns the
resulting state. Idempotent.

---

## `GET /api/server`

Current state. The GUI polls this.

```json
{
  "status": "running",
  "running": true,
  "starting": false,
  "error": null,
  "host": "127.0.0.1",
  "port": 8888,
  "url": "http://127.0.0.1:8888",
  "api_base": "http://127.0.0.1:8888/v1",
  "model_id": "llama-3",
  "uptime": 143.2,
  "command": "/…/gguf-server --model /…/llama-3.gguf --host 127.0.0.1 …",
  "log_path": "/tmp/gguf-server-ab12cd.log"
}
```

`status` is one of `stopped`, `starting`, `running`, `error`. When nothing
has been started, the payload is the stopped shape with empty `url` /
`api_base` and `log_path: null`.

A process that dies while `running` is noticed on the next poll and flipped
to `error` with the log tail folded into the message.

`host` `0.0.0.0` or `::` is reported as `127.0.0.1` in `url` and `api_base`,
because that is the address that actually connects.

---

## `GET /api/log`

```
GET /api/log?max_bytes=200000
```

```json
{ "log": "…last max_bytes of the engine log…" }
```

Tails the file by offset rather than buffering in memory — the engine is
chatty and long-lived. Returns `{"log": ""}` when nothing is running.

---

## Notes for API consumers

- **One server at a time.** There is no handle: `/api/start` always replaces.
- **No authentication.** Anything that can reach the port can start a process
  with arguments of its choosing. Keep it on `127.0.0.1`.
- **Paths are server-side.** The browser never sends file contents; every
  path in a request is resolved on the machine running the panel.
- **Cleanup.** An `atexit` hook stops the managed process, so a clean
  interpreter exit does not orphan the engine.
