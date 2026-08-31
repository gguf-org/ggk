# The unified GUI

```bash
ggk
```

One HTTP server, three panels, one port.

```
┌─ ggk v0.5.1 ──── Server │ Diffuser │ Editor ─────────────────┐
│                                                              │
│   <iframe>  the active panel's own GUI                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## How it is put together

The shell page (`src/ggk/static/index.html`) is chrome only: a header, three
tab buttons and three `<iframe>`s. Panels live under `./server/`,
`./diffuser/` and `./editor/` on the same origin.

Iframes are **lazy and sticky**: a panel's URL is assigned on first
activation and then kept alive, so switching tabs never loses panel state —
a running generation keeps running, a half-edited GGUF file keeps its edits.
The active panel is mirrored into the URL hash (`#diffuser`), so a reload
returns to the same tab.

## Request dispatch

`src/ggk/gui.py` mounts each panel by rewriting the request path and handing
the request to that panel's own `BaseHTTPRequestHandler` subclass:

```
GET /diffuser/api/status
    → strip "/diffuser"
    → self.__class__ = diffuser.server.Handler   (temporarily)
    → do_GET()  sees  /api/status
    → self.__class__ restored
```

The `__class__` swap (rather than delegating a single method) means every
helper method the panel handler calls — `_json`, `_serve_static`, `_error` —
resolves on the right class. A request for `/server` without the trailing
slash gets a `307` to `/server/` so the panel's relative URLs resolve.

The panel frontends build their API URLs relative to the page:

```js
const API_ROOT = new URL('.', location.href).pathname;
function apiPath(p) { return p.startsWith('/') ? API_ROOT + p.slice(1) : p; }
```

so the same `app.js` works mounted at `/diffuser/` here and at `/` when the
panel runs standalone.

## Ports

| | |
| --- | --- |
| Unified GUI | `8640` (`--port`, `0` = auto) |
| Server panel standalone | `8642` |
| Diffuser panel standalone | `8643` |
| Editor panel standalone | `8644` |

The unified GUI does not proxy the LLM engine — that binds its own port
(default `8888`) and is reached directly by API clients.

## Shutdown

Ctrl-C raises `KeyboardInterrupt` in `serve_forever()`. `SIGTERM` (a `kill`,
a closing terminal, a service manager) is caught too and calls
`httpd.shutdown()` from a helper thread, because `shutdown()` must not run on
the serving thread. Either path then calls `gui.shutdown_engines()`, which
stops a managed `gguf-server` child so it is not orphaned.

Diffusion jobs and editor jobs are not stopped on shutdown — they are
short-lived children of the same process and die with it.

## Theming

Each panel carries its own stylesheet and its own light/dark toggle, stored
per panel in `localStorage`:

| Key | Panel |
| --- | ----- |
| `gguf-server-web-theme` | Server |
| `gguf-diffusion-web-theme` | Diffuser |
| `gguf-editor-theme` | Editor |

The shell chrome is always dark.

## Local state

Nothing is stored server-side. Panel configuration lives in the browser:

| Key | What |
| --- | ---- |
| `gguf-server-web-v1` | Current server configuration |
| `gguf-server-web-presets` | Saved server presets |
| `gguf-diffusion-web-v1` | Current compose configuration |
| `gguf-diffusion-web-workflows` | Saved diffusion workflows |
| `gguf-diffusion-web-images` | Recent output gallery (last 50 paths) |

Clearing site data for `127.0.0.1:8640` resets all of it. Because the key
space is per-origin, running the unified GUI and a standalone panel on
different ports gives them separate settings.

## Security posture

The GUI binds `127.0.0.1` by default and is not authenticated. It exposes:

- `POST /api/browse` — directory listings anywhere the process can read
- `POST /api/start` — spawning the engine with configured arguments
- `POST /api/upload` (editor) — writing into a temp directory

Treat `--host 0.0.0.0` as "give everyone on the network a shell-adjacent
capability on this machine", and don't do it on an untrusted network. The
LLM engine's own `--host` / `--api-key` are separate settings; exposing the
inference API is a much smaller step than exposing the control GUI.
