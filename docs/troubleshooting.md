# Troubleshooting

## Install and build

### `engine binary not found at …/bin/gguf-server`

The C/C++ engine did not build during install. Reinstall with output visible:

```bash
pip install --no-cache-dir --force-reinstall -v ggk
```

and read the CMake output. Usual causes: no C/C++ compiler, CMake older than
3.15, or a GPU toolchain that was requested but is not installed.

The same message with `diffusion` or `libgguf_quantizer` means the same
thing for the other two panels.

### `FileNotFoundError: ...\wheel\scripts` at the end of a Windows build

You used `python -m build` without `--wheel`. The compile succeeded; the
packaging step then could not find the temp directory it built in. Use:

```bat
python -m build --wheel
```

See [install.md § Building a wheel](install.md#building-a-wheel).

### CUDA build fails on Windows

`nvcc` only accepts MSVC as a host compiler on Windows. Run from a
`vcvars64` shell and use Ninja:

```bat
call "…\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_ARGS=-G Ninja -DGGK_CUDA=ON
python -m build --wheel
```

### The build takes hours

A CUDA build compiles kernels for every target architecture. That is normal.
Narrow it:

```bash
CMAKE_ARGS='-DGGK_CUDA=ON -DGK_CUDA_ARCHITECTURES="89"' pip install .
```

and always use `--wheel` so a retry is incremental.

### OpenSSL errors on Windows/MSVC

A MinGW/MSYS2 OpenSSL on `PATH` is deliberately ignored, because its headers
cannot be compiled by MSVC. `GGK_OPENSSL` is off by default in `ggk` builds
precisely so this never comes up — if you turned it on, either install an
MSVC OpenSSL (`vcpkg install openssl:x64-windows`, then
`-DOPENSSL_ROOT_DIR=…`) or turn it back off.

### `no kernel image is available for execution on the device`

The binary carries no code for that card. It happens when the architecture
list was pinned too narrowly, or when a wheel built elsewhere was copied onto
a machine with a newer GPU. Rebuild with the architecture listed:

```bash
CMAKE_ARGS='-DGGK_CUDA=ON -DGK_CUDA_ARCHITECTURES="75;86;89;120"' pip install .
```

Left empty, gk works the list out from `nvcc` and the driver, and embeds PTX
for the newest — so this is almost always a pinned list, not the default.

---

## Server panel

### `gguf-server exited (code N)` right after Start

Look at the **Logs** tab; the last few lines are also folded into the error
message. The three usual causes, in order:

1. **Not enough memory.** Lower GPU Layers (0 for CPU-only), or Context
   Length, or use a smaller quantization.
2. **The model file is not what the engine expects.** A corrupt download, or
   a format the runtime does not recognise. Open it in the Editor panel and
   check `general.architecture`.
3. **A bad flag combination** — usually via the *Edit manually* box.

### `did not start within 120s`

The engine is still loading. A very large model on a slow disk, or one that
does not fit and is being paged, can exceed the timeout. The process is not
killed — check the Logs tab, and try again with fewer GPU layers or a smaller
context.

### `Address already in use`

Something is already on the port. Change **Settings → Network → Port**, or
stop the other process. The panel already waits for a previous server *of its
own* to release the port before starting a new one, so this is another
program holding it.

### The GUI says running, but clients get connection refused

Check the host. `127.0.0.1` is unreachable from another machine — bind
`0.0.0.0` for that, and set an API key when you do.

### `401 Unauthorized`

An API key is set in **Settings → Network**. Send it:

```
Authorization: Bearer YOUR_KEY
```

### The model loads but the output is nonsense

Almost always the chat template. Try **Model → Chat Template → custom** with
a built-in name that matches the model family
([server/models.md](server/models.md#chat-templates)). A model whose GGUF
metadata carries a broken `tokenizer.chat_template` will produce fluent
nonsense with the wrong one.

### Tool calls are not parsed

Same cause. The tool-call dialect comes from the chat template; override it
with `--chat-template`. `--skip-chat-parsing` forces everything into
`message.content` if you would rather parse it yourself.

### Very slow despite a GPU

1. Check the device banner on stderr — is the GPU there at all?
2. Check GPU Layers is not 0.
3. Check the model is offloaded: `GET /props`, or watch VRAM in the Hardware
   tab.
4. IQ1/IQ2/IQ3 quantizations decode on the CPU on every backend, so they run
   far slower than a `q4_k` of the same size.

---

## Diffuser panel

### `Process finished but no output file was found`

The engine exited 0 but wrote nothing. Check the Logs tab — usually a model
file the engine could not interpret, or a missing companion file (VAE, text
encoder) that made it bail after loading.

### `Failed to load model` / tensor name errors

The wrong `--model` vs `--diffusion-model` choice, or a missing text encoder.
[diffuser/models.md](diffuser/models.md#what-each-family-needs) lists what
each family needs. Run with `-v` for the tensor names the loader saw.

### Out of memory

In order: `--diffusion-fa`, `--offload-to-cpu`, `--vae-tiling`, a smaller
quantization, `--max-vram N --stream-layers`, `--backend te=cpu,vae=cpu`.
Video also wants `--temporal-tiling`.

### Black or garbage images

- **Wrong VAE.** The latent format must match the model. Try
  `--vae-format flux|sd3|flux2` to override detection.
- **CFG on a distilled model.** Flux, Schnell, Turbo, LCM and friends need
  `--cfg-scale 1.0`. Anything higher both doubles the compute and wrecks the
  output.
- **Too few steps for the sampler.** `dpm++2m` at 4 steps will not work;
  `lcm` or `euler` will.

### The progress bar sticks at the first step

Fixed — the log collapses redraws of the *same* step but keeps a new step as
a new line, so the polling cursor advances. If you see it on an old install,
upgrade.

### `unknown sampling method` / `unknown schedule`

The name is validated against the engine's tables before spawning. Check
spelling against [diffuser/engine.md](diffuser/engine.md#sampling); note
`dpm++2m`, not `dpm++_2m`.

### A tensor split does nothing, or splits backwards

You used `nvidia-smi` device numbers. Use the engine's:

```bash
ggk diffuser engine -- --list-devices
```

See [hardware.md § Two different device lists](hardware.md#two-different-device-lists).

---

## Editor panel

### `Invalid GGUF file (bad magic bytes)`

Not a GGUF file. A safetensors file opens too, but a `.bin`, `.pt`, `.ckpt`
or a half-finished download does not.

### `File appears truncated or malformed`

The header ran past the end of the file — an interrupted download, or a
split-GGUF part that is not the first shard.

### `GGUF header exceeds 1024 MB`

The parser grows its read window up to 1 GiB. A header that large is
malformed.

### `Row size N is not divisible by block size B`

The target quantization type cannot represent that tensor's shape. Pick a
type with a smaller block size — `q8_0` (32) instead of `q4_k` (256) — or
leave that tensor alone. See
[editor/gguf-format.md § Tensor types](editor/gguf-format.md#tensor-types).

### `Duplicate tensor name "…" — rename before saving`

Two tensors would end up with the same name, usually after a find & replace
or an import. Nothing is written until it is resolved.

### The quantizer is unavailable

The banner under the stats bar gives the reason. Usually the library did not
build; reinstall verbosely. On Windows with a MinGW-built DLL, Python 3.8+
resolves dependencies without consulting `PATH` — the loader already retries
with the legacy search, but a genuinely missing runtime DLL will still fail.

### Saving is slow

Saving rewrites the whole file: header plus every tensor's bytes, streamed in
8 MiB chunks. A 70 GB model takes as long as copying 70 GB. Metadata-only
edits are not cheaper, because the header length changes and every tensor
offset shifts with it.

### Drag & drop is slow or fills the disk

A dropped file is **copied** into a temp directory, because a browser cannot
reveal its path. Use the file picker for anything large — it hands over a
path and copies nothing.

---

## GUI

### The browser does not open

Use the URL printed on startup. `--no-browser` skips the attempt entirely;
headless machines and some terminal multiplexers have no browser to open.

### A panel tab is blank

Check the terminal for a traceback. A panel whose engine failed to build
still loads its GUI and shows the error in place — a truly blank frame is
usually a stale cached page; hard-reload.

### Port already in use

Every GUI takes `--port`, and `--port 0` picks a free one:

```bash
ggk --port 0
```

### Settings vanished

Panel configuration lives in the browser's `localStorage`, keyed by origin.
Clearing site data resets it, and the unified GUI (`:8640`) and a standalone
panel (`:8642`) have separate stores. See [gui.md § Local state](gui.md#local-state).

### An engine keeps running after I close the GUI

The server panel stops its child on shutdown, on both Ctrl-C and `SIGTERM`.
A `SIGKILL` of the panel skips that. Find and stop it:

```bash
pkill -f gguf-server
```

Diffusion jobs are short-lived children and die with the panel.

---

## Getting more detail

```bash
# engine-level logging
ggk server engine -- --model model.gguf --verbose
ggk diffuser engine -- … -v

# what devices the build actually has
ggk diffuser engine -- --list-devices
ggk editor devices

# gk's own diagnostics
GK_SCHED_REPORT=1 ggk server engine -- --model model.gguf
GK_OP_PROFILE=1   ggk diffuser engine -- … -v
```

The panel backends print tracebacks to their own stderr, so run them in the
foreground when something is failing.
