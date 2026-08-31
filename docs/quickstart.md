# Quickstart

Everything below assumes `pip install ggk` has finished. See
[install.md](install.md) if it has not.

## 1. Launch the unified GUI

```bash
ggk
```

```
ggk 0.5.1 — serving on http://127.0.0.1:8640/
  server    http://127.0.0.1:8640/server/
  diffuser  http://127.0.0.1:8640/diffuser/
  editor    http://127.0.0.1:8640/editor/
```

A browser opens on the shell page with three tabs. Each tab is the panel's
own GUI, mounted under a prefix on the same port. `--no-browser` skips the
browser, `--port 0` picks a free port.

Ctrl-C (or `SIGTERM`) shuts the GUI down and stops any engine child it
started.

## 2. Run an LLM

In the **Server** tab:

1. Go to **Model** and click *Choose File…* Pick a `.gguf` language model.
   Nothing is uploaded — the picker lists your own filesystem and the engine
   opens the file by path.
2. Go to **Settings** if you want to change context length, GPU layers or the
   port. The **Recommended** button on the Performance card fills in sensible
   values derived from the hardware it detects.
3. Back on **Server**, click **Start**.

Once the status card turns green the OpenAI-compatible API is live at
`http://127.0.0.1:8888/v1`:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hi in five words."}]}'
```

Point any OpenAI client at that base URL. The API key is whatever you set in
**Settings → Network** (empty means no auth).

Full details: [server/README.md](server/README.md).

## 3. Generate an image

In the **Diffuser** tab:

1. Under **Model**, pick the diffusion weights. Choose `--diffusion-model`
   for a bare transformer/UNet file and `--model` for an all-in-one
   checkpoint.
2. Add whatever the family needs — a **VAE**, and text encoders such as
   `--clip_l`, `--t5xxl` or `--llm`. [diffuser/models.md](diffuser/models.md)
   lists what each family expects.
3. Type a **Prompt**, set **Steps** and **CFG Scale**, choose a size.
4. Click **Generate**. Progress and the engine log stream live; finished
   images land in the **Output** tab and on disk in the output directory
   (default `~/Downloads`, named `gguf-YYYYmmdd-HHMMSS.png`).

Full details: [diffuser/README.md](diffuser/README.md).

## 4. Inspect and quantize a GGUF file

In the **Editor** tab:

1. Click *Choose File…* — or drag a `.gguf` / `.safetensors` file anywhere on
   the page — to open it. Only the header is parsed; tensor data stays on
   disk.
2. Edit metadata values in place, rename or delete tensors, reorder rows.
3. Open the **Quantization** drawer, pick a batch weight type (say `q4_k`)
   and/or per-tensor precisions.
4. Click **Save**. With no precision changes the file is rebuilt with your
   edits; with precision changes the edits are applied first and the
   quantizer then writes one output file.

Full details: [editor/README.md](editor/README.md).

## Running one panel on its own

Each panel runs standalone on its own port, exactly as the separate
`gguf-server` / `gguf-diffusion` / `gguf-editor` packages did:

```bash
ggk server            # http://127.0.0.1:8642/
ggk diffuser          # http://127.0.0.1:8643/
ggk editor            # http://127.0.0.1:8644/
ggk editor model.gguf # open a file straight away
```

## Skipping the GUI entirely

The engines are directly scriptable:

```bash
# LLM server on port 8888
ggk server engine -- --model model.gguf --port 8888

# one image
ggk diffuser engine -- -m sd.gguf -p "a lighthouse at dusk" --steps 20 -o out.png

# quantize without opening anything
ggk editor quantize -m in.gguf -o out-q4_k.gguf --type q4_k

# what the quantizer can run on
ggk editor devices
```

Everything after `--` is passed to the engine binary verbatim, so
`ggk diffuser engine -- --help` prints the engine's own help.

Full CLI reference: [cli.md](cli.md).
