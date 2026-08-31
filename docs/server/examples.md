# Server panel — examples

All examples assume the server is running on `http://127.0.0.1:8888`, either
started from the GUI or with `ggk server engine`.

## Starting the engine

```bash
# minimal
ggk server engine -- --model model.gguf

# a typical GPU setup
ggk server engine -- \
    --model model.gguf \
    --ctx-size 16384 \
    --n-gpu-layers 999 \
    --flash-attn on \
    --parallel 4 \
    --port 8888

# CPU-only, memory-tight
ggk server engine -- \
    --model model.gguf \
    --n-gpu-layers 0 \
    --ctx-size 4096 \
    --batch-size 512 --ubatch-size 256 \
    --threads 8

# quantized KV cache to fit a longer context
ggk server engine -- \
    --model model.gguf --ctx-size 131072 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
```

## Chat completions

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

Streaming — server-sent events, same shape as OpenAI:

```bash
curl -N http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to ten."}],"stream":true}'
```

## With the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="model",                       # or your --alias
    messages=[{"role": "user", "content": "Write a haiku about GGUF."}],
)
print(resp.choices[0].message.content)
```

If you set an API key in **Settings → Network**, pass it as `api_key`.

## With the Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8888", api_key="not-needed")

msg = client.messages.create(
    model="model",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(msg.content[0].text)
```

## Text completions (non-chat)

```bash
curl http://127.0.0.1:8888/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "def fibonacci(n):", "max_tokens": 128, "temperature": 0.2}'
```

The engine's native `/completion` endpoint exposes more sampling controls
than the OpenAI shape allows — grammars, logit bias, mirostat, per-request
`n_probs`. See `vendor/engine/app/README.md`.

## Embeddings

Start the server restricted to embeddings:

```bash
ggk server engine -- --model nomic-embed-text.gguf --embeddings --pooling mean
```

```bash
curl http://127.0.0.1:8888/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["the first document", "the second document"]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="x")
vecs = client.embeddings.create(model="emb", input=["hello", "world"])
print(len(vecs.data[0].embedding))
```

## Reranking

```bash
ggk server engine -- --model bge-reranker.gguf --rerank
```

```bash
curl http://127.0.0.1:8888/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "how do I quantize a model?",
    "documents": [
      "Quantization reduces weight precision.",
      "The capital of France is Paris.",
      "Use ggk editor quantize with --type q4_k."
    ],
    "top_n": 2
  }'
```

## Vision

```bash
ggk server engine -- --model qwen2-vl.gguf --mmproj mmproj-qwen2-vl.gguf
```

```python
import base64, openai

img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
client = openai.OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="x")

resp = client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

## Structured output with a JSON schema

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Invent a city."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "city",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "population": {"type": "integer"},
            "founded": {"type": "integer"}
          },
          "required": ["name", "population", "founded"]
        }
      }
    }
  }'
```

The schema is compiled to a grammar and constrains decoding, so the output is
always parseable.

## Tool calling

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Oslo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

Tool-call parsing follows the model's chat template. If a model emits tool
calls in a dialect the template does not describe, `--chat-template` with a
matching built-in usually fixes it.

## Reasoning models

```bash
ggk server engine -- --model deepseek-r1.gguf \
    --reasoning on --reasoning-format deepseek --reasoning-budget 2048
```

`--reasoning-format deepseek` puts the thinking trace in
`message.reasoning_content` and leaves `message.content` clean.
`--reasoning-budget` caps how long the model may think before being nudged to
answer.

## Router mode — several models on one port

```bash
ggk server engine -- --models-dir ~/models --models-autoload --models-max 2
```

```bash
curl http://127.0.0.1:8888/models                   # what's available
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-3-8b-q4_k", "messages": [{"role":"user","content":"hi"}]}'
```

The parent process loads models on demand, evicting to stay within
`--models-max`. In the GUI this needs the **Edit manually** command box.

## Health, properties and metrics

```bash
curl http://127.0.0.1:8888/health
curl http://127.0.0.1:8888/props     | python3 -m json.tool
curl http://127.0.0.1:8888/slots     | python3 -m json.tool
curl http://127.0.0.1:8888/metrics                    # needs --metrics
```

## Driving the panel itself

The panel's control API is scriptable too — useful for a headless box:

```bash
# start the panel without a browser
ggk server --no-browser --port 8642 &

# start an engine through it
curl -s http://127.0.0.1:8642/api/start \
  -H 'Content-Type: application/json' \
  -d '{"config": {"model_path": "/models/llama-3.gguf", "port": 8888,
                  "context_length": 8192, "gpu_layers": 999}}'

# poll until running
until curl -s http://127.0.0.1:8642/api/server | grep -q '"running": true'; do sleep 1; done

# ... use the model ...

curl -s -X POST http://127.0.0.1:8642/api/stop
```

Full schemas: [api.md](api.md).
