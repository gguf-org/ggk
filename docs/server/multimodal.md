# Multimodal — images and audio

A multimodal model runs as **two GGUF files**:

1. the language model itself, and
2. a **multimodal projector** (`mmproj`) that encodes images or audio into
   embeddings the language model can consume.

Keeping the projector separate is deliberate: pre-processing and projection
vary wildly between vision models, and folding that into the core runtime
would make every one of them the runtime's problem.

## Using one

In the GUI: **Model** tab → pick the model file, then pick the projector in
the *Multimodal projector* field.

On the command line:

```bash
ggk server engine -- --model model.gguf --mmproj mmproj.gguf
```

Then send images the OpenAI way:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}}
      ]
    }]
  }'
```

`--media-path DIR` additionally allows `file://` URLs relative to that
directory, which avoids base64-ing large files.

## Flags

| Flag | Meaning |
| ---- | ------- |
| `-mm, --mmproj FILE` | The projector file |
| `--mmproj-offload`, `--no-mmproj-offload` | GPU-offload the projector (on by default) |
| `--mmproj-auto`, `--no-mmproj-auto` | Use a projector automatically when one is found next to the model |
| `--image-min-tokens N` | Minimum tokens per image (dynamic-resolution models) |
| `--image-max-tokens N` | Maximum tokens per image |
| `--mtmd-batch-max-tokens N` | Image tokens per encode batch (default 1024) |
| `--media-path PATH` | Directory `file://` media references resolve against |

Image tokens count against the context window. A high-resolution image on a
dynamic-resolution model can consume thousands of tokens; `--image-max-tokens`
is the knob that bounds it.

## Projector types supported

The runtime dispatches on the projector's own type field. Vision projectors:

`mlp` · `mlp_norm` · `ldp` · `ldpv2` · `resampler` (MiniCPM-V) ·
`adapter` (GLM-Edge) · `qwen2vl` · `qwen25vl` · `qwen3vl` · `gemma3` ·
`gemma3n` · `gemma4v` · `gemma4uv` · `idefics3` · `pixtral` · `internvl` ·
`llama4` · `phi4` · `lfm2` · `kimivl` · `kimik25` · `cogvlm` · `janus_pro` ·
`glm4v` · `youtuvl` · `step3vl` · `mimovl` · `minimax_m3` ·
`nemotron_v2_vl` · `hunyuanvl` · `exaone4_5` · `minicpmv4_6` ·
`granite4_vision` · `muse_glimmer`

OCR-specialised projectors: `paddleocr` · `lightonocr` · `dots_ocr` ·
`deepseekocr` · `deepseekocr2`

Audio projectors: `ultravox` · `qwen2a` · `qwen3a` · `glma` · `voxtral` ·
`meralion` · `music_flamingo` · `lfm2a` · `granite_speech` · `yasa2` ·
`qwen25o` (omni; resolves to the vision or audio path per context)

## Models that convert cleanly

For these, upstream `convert_hf_to_gguf.py --mmproj` produces the projector
directly from the Hugging Face checkpoint:

- **Gemma 3** (the 1B variant has no vision tower)
- **SmolVLM** and **SmolVLM2**
- **Pixtral 12B** — the `transformers`-compatible checkpoint only
- **Qwen 2-VL** and **Qwen 2.5-VL**
- **Mistral Small 3.1 24B**
- **InternVL 2.5** and **InternVL 3** — the non-HF variants
- **MiniCPM-V 4.6** — needs a `transformers` 5.7.0+ checkpoint

Older families (LLaVA, MobileVLM, GLM-Edge, MiniCPM-V 2.5 / 2.6 / 4.x,
MiniCPM-o, IBM Granite Vision) have their own legacy conversion scripts; see
the upstream `llama.cpp` multimodal docs for those.

Most people never convert anything — pre-quantized model + `mmproj` pairs are
published on Hugging Face for every popular vision model.

## Practical notes

- **The pair must match.** A projector built for one model revision will
  produce nonsense, or fail to load, against another.
- **Projector on GPU.** `--mmproj-offload` is on by default and is usually
  what you want; turning it off frees a little VRAM at a large latency cost
  per image.
- **The model must be vision-capable.** Attaching a projector to a text-only
  model does nothing useful — the language model needs the matching
  cross-modal tokens in its vocabulary and graph.
- **Audio input** works the same way, with an audio projector; send audio
  parts in the message content.

## Where the code lives

`vendor/engine/mtmd/` — `clip.cpp` for the vision/audio towers, `mtmd.cpp`
for the unified interface, `mtmd-audio.cpp` for audio pre-processing.
`vendor/engine/mtmd/README.md` has the upstream background.
