# Supported models — server panel

The server panel loads **GGUF** language models. The architecture is read from
the file's `general.architecture` metadata key; if the runtime has a graph
builder for it, the model runs.

To see what a file claims to be without loading it, open it in the
[Editor panel](../editor/README.md) and look at `general.architecture`.

## Architectures

Every value below is one the runtime recognises. Names are exactly the
strings written into GGUF metadata.

### Llama family and derivatives

`llama` · `llama4` · `llama-embed` · `deci` · `mistral3` · `mistral4` ·
`arcee` · `smollm3` · `openelm` · `xverse` · `yi` (as `llama`) ·
`eagle3` · `dflash`

### Qwen

`qwen` · `qwen2` · `qwen2moe` · `qwen2vl` · `qwen3` · `qwen3moe` ·
`qwen3next` · `qwen3vl` · `qwen3vlmoe` · `qwen35` · `qwen35moe`

### Gemma

`gemma` · `gemma2` · `gemma3` · `gemma3n` · `gemma4` · `gemma4-assistant` ·
`gemma-embedding`

### DeepSeek and GLM

`deepseek` · `deepseek2` · `deepseek2-ocr` · `deepseek32` · `deepseek4` ·
`chatglm` · `glm4` · `glm4moe` · `glm-dsa`

### Phi, Granite, Nemotron, Command-R

`phi2` · `phi3` · `phimoe` ·
`granite` · `granitemoe` · `granitehybrid` ·
`nemotron` · `nemotron_h` · `nemotron_h_moe` ·
`command-r` · `cohere2` · `cohere2moe`

### Mixture-of-experts and large open models

`grok` · `dbrx` · `arctic` · `jamba` · `bailingmoe` · `bailingmoe2` ·
`dots1` · `grovemoe` · `afmoe` · `minimax-m2` · `minimax-m3` ·
`hunyuan-moe` · `hunyuan-dense` · `hy_v3` · `ernie4_5` · `ernie4_5-moe` ·
`gpt-oss` · `apertus` · `seed_oss` · `kimi-linear` · `step35`

### Recurrent / hybrid / state-space

`mamba` · `mamba2` · `rwkv6` · `rwkv6qwen2` · `rwkv7` · `arwkv7` ·
`falcon-h1` · `lfm2` · `lfm2moe` · `plamo` · `plamo2` · `plamo3` ·
`llada` · `llada-moe` · `dream` · `rnd1`

### Embedding and reranking

`bert` · `modern-bert` · `neo-bert` · `nomic-bert` · `nomic-bert-moe` ·
`jina-bert-v2` · `jina-bert-v3` · `eurobert` · `gemma-embedding` ·
`llama-embed` · `t5` · `t5encoder`

### Vision / OCR / multimodal text towers

`clip` · `qwen2vl` · `qwen3vl` · `qwen3vlmoe` · `cogvlm` · `hunyuan_vl` ·
`paddleocr` · `deepseek2-ocr` · `mimo2` · `muse-glimmer`

(These need an `mmproj` companion — see [multimodal.md](multimodal.md).)

### Code and specialised

`starcoder` · `starcoder2` · `codeshell` · `refact` · `maincoder` ·
`mellum` · `smallthinker` · `talkie` · `wavtokenizer-dec` · `bitnet`

### Older / smaller architectures

`gpt2` · `gptj` · `gptneox` · `mpt` · `falcon` · `baichuan` · `bloom` ·
`stablelm` · `orion` · `internlm2` · `minicpm` · `minicpm3` · `olmo` ·
`olmo2` · `olmoe` · `chameleon` · `exaone` · `exaone4` · `exaone-moe` ·
`jais` · `jais2` · `plm` · `laguna` · `pangu-embedded` · `nanbeige`

> This list tracks `LLM_ARCH_*` in `vendor/engine/src/llama-arch.cpp`. If a
> new model lands upstream and you rebuild, it appears there first.

## Weight types the runtime can read

Any GGUF quantization the decoder implements:

| Class | Types |
| ----- | ----- |
| Float | `F32`, `F16`, `BF16`, `F64` |
| Legacy | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` |
| I-quants | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` |
| Ternary | `TQ1_0`, `TQ2_0` |
| FP4/FP8-style | `MXFP4`, `NVFP4` |
| New small blocks | `Q1_0`, `Q2_0` |
| Integer | `I8`, `I16`, `I32`, `I64` |

The same table drives the Editor panel's quantizer — see
[../editor/quantizer.md](../editor/quantizer.md).

### Picking a quantization

| Type | Bits/weight (approx.) | When |
| ---- | --------------------- | ---- |
| `Q8_0` | 8.5 | Near-lossless; use when memory is not the constraint |
| `Q6_K` | 6.6 | Very close to `Q8_0` at 3/4 the size |
| `Q5_K` | 5.5 | Good default for quality-sensitive work |
| `Q4_K` | 4.5 | The usual sweet spot for local use |
| `IQ4_XS` | 4.25 | Slightly smaller than `Q4_K`, needs more compute |
| `Q3_K` | 3.4 | Noticeable degradation; for models that barely fit |
| `IQ2_*`, `IQ1_*` | 1–2.5 | Only for very large models where nothing else fits |

## Chat templates

`--jinja` is on by default and the model's own template (from
`tokenizer.chat_template`) is used. When a model ships no template, or ships
a broken one, name a built-in with `--chat-template` or the GUI's **Model →
Chat Template** control:

```
bailing · bailing-think · bailing2 · chatglm3 · chatglm4 · chatml ·
command-r · deepseek · deepseek-ocr · deepseek2 · deepseek3 · exaone-moe ·
exaone3 · exaone4 · falcon3 · gemma · gigachat · glmedge · gpt-oss ·
granite · granite-4.0 · granite-4.1 · grok-2 · hunyuan-dense · hunyuan-moe ·
hunyuan-vl · kimi-k2 · llama2 · llama2-sys · llama2-sys-bos ·
llama2-sys-strip · llama3 · llama4 · megrez · minicpm · mistral-v1 ·
mistral-v3 · mistral-v3-tekken · mistral-v7 · mistral-v7-tekken · monarch ·
openchat · orion · pangu-embedded · phi3 · phi4 · rwkv-world · seed_oss ·
smolvlm · solar-open · vicuna · vicuna-orca · yandex · zephyr
```

An arbitrary Jinja template can be passed inline (`--chat-template '…'`) or
from a file (`--chat-template-file tmpl.jinja`). The GUI's template file
picker accepts `.json`, `.jinja`, `.jinja2` and `.txt`.

## Embedding and reranking models

Embedding models want the server restricted to that use:

```bash
ggk server engine -- --model nomic-embed.gguf --embeddings --pooling mean
```

`--pooling` takes `none`, `mean`, `cls`, `last` or `rank`; leaving it unset
uses the model's own default. `--embd-normalize N` controls normalisation
(`-1` none, `0` max-abs int16, `1` taxicab, `2` euclidean — the default,
`>2` p-norm).

Reranking models need `--rerank`, which enables `POST /v1/rerank` and
`POST /reranking`.

## Getting models

The panels take local file paths — download however you like. The engine
itself can fetch from Hugging Face when built with OpenSSL
(`GGK_OPENSSL=1`), via `-hf <user>/<repo>[:quant]`, caching under
`LLAMA_CACHE`. `ggk` builds ship without OpenSSL by default, so the normal
path is to download with your own tool and point the GUI at the file.
