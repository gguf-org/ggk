# Diffuser panel — examples

Every example is a `ggk diffuser engine` invocation. The same settings can be
entered in the Compose tab; the **Command** card shows exactly this text.

## Text to image

### SD 1.5 (all-in-one checkpoint)

```bash
ggk diffuser engine -- \
    -m sd-v1-5-q8_0.gguf \
    -p "a watercolor fox in a pine forest" \
    -n "blurry, low quality" \
    --cfg-scale 7.0 --steps 25 --sampling-method euler_a \
    -W 512 -H 512 -o fox.png
```

### SDXL

```bash
ggk diffuser engine -- \
    --diffusion-model sdxl-base-q4_k.gguf \
    --vae sdxl-vae-fp16.safetensors \
    --clip_l clip_l.safetensors --clip_g clip_g.safetensors \
    -p "product photo of a ceramic mug on marble, studio lighting" \
    -n "text, watermark" \
    --cfg-scale 7.0 --steps 30 --sampling-method dpm++2m --scheduler karras \
    -W 1024 -H 1024 -o mug.png
```

### Flux Dev

```bash
ggk diffuser engine -- \
    --diffusion-model flux1-dev-q4_k.gguf \
    --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    -p "a lighthouse at dusk, long exposure, 35mm" \
    --cfg-scale 1.0 --guidance 3.5 --steps 20 \
    --sampling-method euler --diffusion-fa \
    -W 1024 -H 1024 -o lighthouse.png
```

Flux is distilled: keep `--cfg-scale 1.0` and steer with `--guidance`. A
negative prompt does nothing here.

### Flux Schnell — four steps

```bash
ggk diffuser engine -- \
    --diffusion-model flux1-schnell-q4_k.gguf \
    --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    -p "a neon-lit ramen shop in the rain" \
    --cfg-scale 1.0 --steps 4 -W 1024 -H 1024 -o ramen.png
```

### SD 3.5 Medium with skip-layer guidance

```bash
ggk diffuser engine -- \
    --diffusion-model sd3.5_medium-q8_0.gguf \
    --clip_l clip_l.safetensors --clip_g clip_g.safetensors \
    --t5xxl t5xxl_fp8.safetensors \
    -p "an isometric diorama of a mountain village" \
    --cfg-scale 4.5 --steps 28 --sampling-method euler \
    --slg-scale 2.5 --skip-layers "7,8,9" \
    -W 1024 -H 1024 -o village.png
```

### Qwen-Image

```bash
ggk diffuser engine -- \
    --diffusion-model qwen-image-q4_k.gguf \
    --vae qwen-image-vae.safetensors \
    --llm qwen2.5-vl-7b-q4_k.gguf \
    -p "a bookshop sign reading OPEN in hand-painted letters" \
    --cfg-scale 4.0 --steps 25 -W 1024 -H 1024 -o shop.png
```

## Image to image

```bash
ggk diffuser engine -- \
    --diffusion-model sdxl-base-q4_k.gguf --vae sdxl-vae.safetensors \
    --clip_l clip_l.safetensors --clip_g clip_g.safetensors \
    -i sketch.png --strength 0.6 \
    -p "finished oil painting, thick impasto" \
    --cfg-scale 7.0 --steps 30 -o painting.png
```

`--strength` is how far to travel from the input: 0.2 barely changes it, 0.9
is nearly a fresh generation.

## Inpainting

```bash
ggk diffuser engine -- \
    --diffusion-model sd-v1-5-inpaint-q8_0.gguf \
    -i room.png --mask mask.png \
    -p "a tall potted fern" \
    --cfg-scale 7.0 --steps 25 --strength 1.0 -o room-fern.png
```

White areas of the mask are repainted, black are kept.

## Flux Kontext — reference-guided editing

```bash
ggk diffuser engine -- \
    --diffusion-model flux1-kontext-q4_k.gguf \
    --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    -r subject.png \
    -p "the same character, now wearing a red scarf, snowy street" \
    --cfg-scale 1.0 --guidance 2.5 --steps 20 -o edited.png
```

Multiple references: repeat `-r`, and add `--increase-ref-index` so they are
numbered in the order given.

## ControlNet

```bash
ggk diffuser engine -- \
    --diffusion-model sd-v1-5-q8_0.gguf \
    --control-net control-canny.safetensors \
    --control-image edges.png --control-strength 0.9 \
    -p "a cathedral interior, volumetric light" \
    --cfg-scale 7.0 --steps 25 -o cathedral.png
```

`--control-strength 1.0` destroys all information from the init image;
0.6–0.9 is the usual range.

## LoRA

LoRAs are referenced from the prompt and resolved against
`--lora-model-dir`:

```bash
ggk diffuser engine -- \
    --diffusion-model flux1-dev-q4_k.gguf --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    --lora-model-dir ./loras \
    -p "a portrait in the style of <lora:my-style:0.8>" \
    --cfg-scale 1.0 --guidance 3.5 --steps 20 -o portrait.png
```

On quantized weights the engine picks `at_runtime` application automatically;
force it either way with `--lora-apply-mode`.

## Batches and seeds

```bash
# four images, random seeds
ggk diffuser engine -- -m model.gguf -p "a paper crane" -b 4 -o crane.png
# → crane_0000.png crane_0001.png crane_0002.png crane_0003.png

# reproduce exactly
ggk diffuser engine -- -m model.gguf -p "a paper crane" -s 12345 -o crane.png
```

## Video — Wan 2.2

```bash
ggk diffuser engine -- --mode vid_gen \
    --diffusion-model wan2.2-ti2v-low-q4_k.gguf \
    --high-noise-diffusion-model wan2.2-ti2v-high-q4_k.gguf \
    --vae wan2.2-vae.safetensors \
    --t5xxl umt5-xxl-q8_0.gguf \
    -p "a hawk banking over a canyon at sunrise" \
    --video-frames 81 --fps 16 \
    --steps 30 --cfg-scale 5.0 --flow-shift 5.0 \
    --moe-boundary 0.875 \
    --vae-tiling --offload-to-cpu \
    -W 832 -H 480 -o hawk.png
```

Video is memory-hungry. `--vae-tiling`, `--temporal-tiling` and
`--offload-to-cpu` are usually the difference between a run and an OOM.

First/last frame interpolation:

```bash
ggk diffuser engine -- --mode vid_gen \
    --diffusion-model wan2.2-i2v-q4_k.gguf --vae wan-vae.safetensors \
    --t5xxl umt5-xxl-q8_0.gguf \
    -i first.png --end-img last.png \
    -p "smooth camera push-in" --video-frames 49 --fps 16 -o clip.png
```

## Audio — ACE-Step

```bash
ggk diffuser engine -- \
    -m ace-step-q8_0.gguf \
    -p "dream pop, reverb-soaked guitars, 90 bpm" \
    --lyrics "we drift through the amber evening" \
    --audio-duration 45 --steps 60 -o song.wav
```

Leave `--lyrics` out for instrumental.

## Upscaling

```bash
ggk diffuser engine -- --mode upscale \
    --upscale-model realesrgan-x4.safetensors \
    -i small.png --upscale-repeats 1 --upscale-tile-size 128 \
    -o large.png
```

Hires fix as a second pass inside one generation:

```bash
ggk diffuser engine -- \
    --diffusion-model sdxl-base-q4_k.gguf --vae sdxl-vae.safetensors \
    --clip_l clip_l.safetensors --clip_g clip_g.safetensors \
    -p "a storm over wheat fields" \
    --steps 30 -W 768 -H 768 \
    --hires --hires-scale 1.5 --hires-steps 12 --hires-denoising-strength 0.55 \
    -o storm.png
```

## Reading an image's metadata

```bash
ggk diffuser engine -- --mode metadata --image storm.png
ggk diffuser engine -- --mode metadata --image storm.png --metadata-format json
```

Metadata is embedded automatically unless `--disable-image-metadata` was set.

## Multi-GPU

```bash
# transformer split across two CUDA devices, VAE on the CPU
ggk diffuser engine -- \
    --diffusion-model flux1-dev-q8_0.gguf --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    --backend "diffusion=cuda0&cuda1,te=cpu,vae=cpu" \
    --split-mode "diffusion=layer" \
    -p "a glass observatory on a cliff" --steps 20 -o obs.png
```

Get the device names from the engine — never from `nvidia-smi`:

```bash
ggk diffuser engine -- --list-devices
```

## Fitting a large model into small VRAM

```bash
ggk diffuser engine -- \
    --diffusion-model flux1-dev-q4_k.gguf --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    -p "a botanical illustration of a fern" \
    --steps 20 -W 1024 -H 1024 \
    --max-vram 6 --stream-layers \
    --offload-to-cpu --vae-tiling --diffusion-fa \
    -o fern.png
```

In rough order of what to try first: `--diffusion-fa`, `--offload-to-cpu`,
`--vae-tiling`, a smaller quantization, `--max-vram` with `--stream-layers`,
`--backend te=cpu`.

## Faster iteration with step caching

```bash
ggk diffuser engine -- \
    --diffusion-model flux1-dev-q4_k.gguf --vae ae.safetensors \
    --clip_l clip_l.safetensors --t5xxl t5xxl_fp8.safetensors \
    -p "an art nouveau poster of a comet" \
    --steps 28 --cache-mode easycache --cache-option "threshold=0.25" \
    -o comet.png
```

## Driving the panel from a script

```bash
ggk diffuser --no-browser --port 8643 &

JOB=$(curl -s http://127.0.0.1:8643/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"config": {"model_path": "/models/sd15.gguf", "model_mode": "model",
                  "prompt": "a paper crane", "steps": 20, "cfg_scale": 7.0,
                  "width": 512, "height": 512, "output_dir": "/tmp/out"}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')

while :; do
  S=$(curl -s "http://127.0.0.1:8643/api/job/$JOB")
  echo "$S" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["status"], d["progress"])'
  echo "$S" | grep -q '"status": "running"' || break
  sleep 1
done
```

Full schemas: [api.md](api.md).
