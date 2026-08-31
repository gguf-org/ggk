# Diffuser panel — local JSON API

The API of the **panel** (the Python control plane). It exists so the web GUI
can start generations, watch them, and read the results back.

Base URL:

- standalone: `http://127.0.0.1:8643/api/…`
- under the unified GUI: `http://127.0.0.1:8640/diffuser/api/…`

Responses are JSON with `Cache-Control: no-store`. Errors are
`{"error": "message"}` with a 4xx/5xx status.

---

## `GET /api/status`

Static facts about this build, plus the option tables the GUI populates its
dropdowns from.

```json
{
  "version": "0.5.1",
  "engine_available": true,
  "engine_error": null,
  "engine_path": "/…/ggk/diffuser/bin/diffusion",
  "sampling_methods": ["euler", "euler_a", "heun", "…"],
  "schedules": ["discrete", "karras", "exponential", "…"],
  "text_encoder_flags": ["--llm", "--clip_l", "--clip_g", "--clip_vision", "--t5xxl", "--llm_vision", "--llm-adapter"],
  "additional_model_flags": ["--high-noise-diffusion-model", "--uncond-diffusion-model", "--control-net", "--photo-maker", "--upscale-model", "--taesd", "--embeddings-connectors", "--audio-vae"],
  "tensor_split_modules": ["diffusion", "te", "clip-vision", "vae", "control-net", "photo-maker", "upscaler"],
  "tensor_split_modes": ["layer", "row"],
  "home": "/Users/you",
  "default_output_dir": "/Users/you/Downloads",
  "windows": false
}
```

The flag allow-lists are enforced server-side: a `text_encoders` or
`additional_models` row naming a flag outside them is silently dropped.

---

## `GET /api/devices`

The compute devices the **engine** can address — the only names `--backend`
accepts.

```json
{ "devices": [
  {"name": "cuda0", "description": "NVIDIA GeForce RTX 5090", "is_gpu": true,
   "memory_free": 30000000000, "memory_total": 34359738368},
  {"name": "cpu", "description": "gk CPU backend", "is_gpu": false}
]}
```

Obtained by running `diffusion --list-devices` once and caching the result.
A device the build has no kernels for is not listed — the engine drops it at
discovery rather than failing partway through a generation.

**This is deliberately separate from `/api/hardware`.** CUDA numbers its
devices fastest-first by default, so on a machine with an RTX 5090 and an RTX
4050 the engine's `cuda0` is the 5090 while `nvidia-smi` calls the 4050 GPU 0.
Guessing gets a tensor split backwards, silently. If the engine is too old
for the flag or fails to start, this returns an empty list and the GUI falls
back to a free-text device box.

---

## `GET /api/hardware`

Best-effort system view: CPU count, RAM (Linux `/proc/meminfo`), and NVIDIA
GPUs from `nvidia-smi`.

```json
{
  "os": "linux",
  "cpu_count": 16,
  "total_ram_mib": 32014,
  "available_ram_mib": 21000,
  "gpus": [{"name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_total_mib": 6140, "vram_used_mib": 512,
            "utilization_percent": 3, "temperature_c": 41}]
}
```

Informational only. Use `/api/devices` for anything that feeds `--backend`.

---

## `POST /api/browse`

Directory listing for the file pickers.

```json
{ "path": "/Users/you/models", "kind": "model" }
```

| `kind` | Shows |
| ------ | ----- |
| `model` | `.gguf`, `.safetensors` |
| `image` | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif` |
| `dir` | directories only |
| `any` | everything non-hidden |

Same response shape as the server panel's:

```json
{
  "path": "/Users/you/models",
  "parent": "/Users/you",
  "entries": [
    {"name": "flux", "path": "/Users/you/models/flux", "is_dir": true},
    {"name": "ae.safetensors", "path": "/Users/you/models/ae.safetensors", "is_dir": false, "size": 335304388}
  ]
}
```

---

## `POST /api/generate`

Start a generation. Returns immediately with a job id.

```json
{
  "config": {
    "model_mode": "diffusion-model",
    "model_path": "/models/flux1-dev-q4_k.gguf",
    "vae_path": "/models/ae.safetensors",
    "tokenizer_pack_path": "",
    "text_encoders": [
      {"flag": "--clip_l", "path": "/models/clip_l.safetensors"},
      {"flag": "--t5xxl",  "path": "/models/t5xxl_fp8.safetensors"}
    ],
    "additional_models": [],
    "init_image_path": "", "mask_image_path": "", "end_image_path": "",
    "control_image_path": "", "ref_image_paths": [],
    "strength": 0.75, "control_strength": 0.9,
    "increase_ref_index": false, "disable_auto_resize_ref_image": false,
    "prompt": "a lighthouse at dusk",
    "negative_prompt": "",
    "cfg_scale": 1.0, "steps": 20,
    "width": 1024, "height": 1024,
    "seed": -1, "batch_count": 1,
    "sampling_method": "euler", "schedule": "discrete",
    "flash_attn": true, "verbose": true,
    "mmap": false, "offload_to_cpu": false,
    "clip_on_cpu": false, "vae_on_cpu": false, "control_net_cpu": false,
    "tensor_split": [{"module": "diffusion", "devices": ["cuda0", "cuda1"], "split_mode": "layer"}],
    "output_dir": "/Users/you/Downloads",
    "use_custom_command": false,
    "custom_command": ""
  }
}
```

```json
{
  "job_id": "a1b2c3d4e5f6",
  "output_path": "/Users/you/Downloads/gguf-20260831-143512.png",
  "command": "/…/diffusion --diffusion-model /models/flux1-dev-q4_k.gguf …"
}
```

The output filename is generated from the timestamp; the output directory is
created if missing. With `use_custom_command: true` the command is tokenised
with `shlex`, a leading binary name is stripped, and the rest of the config is
ignored — including `-o`, so a custom command must supply its own.

`400` for: no model selected, model file not found, a tokenizer pack that is
not an existing directory, an unknown sampling method or schedule, or an
empty custom command.

Width and height are clamped to 64–2048. `seed < 0` omits `--seed` entirely,
so the engine randomises.

---

## `GET /api/job/{id}`

Poll a job. `?after=N` returns only log lines from index `N` onward.

```json
{
  "id": "a1b2c3d4e5f6",
  "status": "running",
  "step": 7,
  "steps": 20,
  "progress": 35,
  "log": ["  |=====>      | 7/20 - 1.23it/s"],
  "log_next": 42,
  "error": null,
  "output_files": [],
  "elapsed": 12.4,
  "command": "/…/diffusion …"
}
```

`status` is `running`, `completed`, `error` or `cancelled`. Feed `log_next`
back as the next `after` to stream without re-fetching.

On completion `output_files` holds every file found: `base.png` for a single
image, or `base_0000.png`, `base_0001.png` … for a batch. A process that
exits with no output file is reported as `error` with the last error-looking
log line appended.

`404` for an unknown job id.

---

## `POST /api/job/{id}/cancel`

```json
{ "ok": true }
```

Sends `SIGTERM`, then `SIGKILL` after 5 s from a helper thread. The job ends
as `cancelled`.

---

## `GET /api/image?path=…`

Streams an image file, 1 MiB at a time. Only files whose suffix is in the
image list are served; anything else is `404`.

```
GET /api/image?path=/Users/you/Downloads/gguf-20260831-143512.png
```

---

## Notes for API consumers

- **Jobs are in-memory and per-process.** Restarting the panel forgets them;
  the files on disk remain.
- **Concurrency is unbounded.** Nothing stops you starting five generations
  at once — the GPU will, eventually, and not politely.
- **No authentication.** Keep it on `127.0.0.1`.
- **Paths are server-side.** Nothing is uploaded; every path is resolved on
  the machine running the panel.
