# Development

## Repository layout

```
ggk/
├── CMakeLists.txt          builds the engine, installs it into the package
├── pyproject.toml          scikit-build-core packaging
├── README.md
├── docs/                   this documentation
├── src/ggk/                the Python package
│   ├── __main__.py         the `ggk` CLI
│   ├── gui.py              the unified 3-panel shell
│   ├── static/             the shell page
│   ├── server/             LLM server panel
│   │   ├── __main__.py     `ggk server`
│   │   ├── server.py       HTTP backend + /api/*
│   │   ├── engine.py       arg building, process supervision, hardware probe
│   │   ├── static/         the web GUI
│   │   └── bin/            gguf-server  (installed by the build)
│   ├── diffuser/           diffusion panel
│   │   ├── __main__.py     `ggk diffuser`
│   │   ├── server.py       HTTP backend + /api/*
│   │   ├── engine.py       arg building, job supervision, device probe
│   │   ├── static/         the web GUI
│   │   └── bin/            diffusion  (installed by the build)
│   └── editor/             GGUF editor panel
│       ├── __main__.py     `ggk editor`
│       ├── server.py       HTTP backend + /api/*, job runner
│       ├── gguf.py         GGUF parser and streaming writer
│       ├── edits.py        applies an edit spec to a parsed file
│       ├── quantizer.py    ctypes bindings
│       ├── static/         the web GUI
│       └── lib/            libgguf_quantizer  (installed by the build)
└── vendor/engine/          the C/C++ engine — see docs/engine/
```

## Design rules the code follows

**The Python side is stdlib only.** No Flask, no FastAPI, no numpy. Every
panel backend is a `BaseHTTPRequestHandler` on a `ThreadingHTTPServer`.
`pip install ggk` therefore pulls in nothing but a build backend, and the
package works in any 3.8+ environment. Keep it that way.

**The frontends are vanilla JS.** No build step, no bundler, no framework.
Each panel is one `index.html`, one `app.js`, one `style.css`, served from
disk with `Cache-Control: no-store` — so editing a frontend is a reload, not
a rebuild.

**Paths, not uploads.** The panel runs on the same machine as the browser, so
files are addressed by filesystem path through `/api/browse`. Uploading a
multi-GB model into temp storage to then hand the engine a path is the thing
the design is avoiding. The editor's drag & drop is the single, deliberate
exception.

**The Python side never touches a tensor.** It builds arguments, spawns
processes, tails logs, and — in the editor — copies bytes. Everything
numerical happens in the engine.

**Panels are independent.** `gui.py` mounts them; it does not couple them.
Each panel's handler class works unchanged at `/` and under a prefix, because
the frontends build API URLs relative to the page.

## Editing the Python side

An editable install picks up changes immediately:

```bash
pip install -e .
ggk --no-browser
```

Restart the process for backend changes; reload the page for frontend
changes.

## Editing the frontends

No build step. Edit `src/ggk/*/static/app.js` and reload — `no-store` means
the browser never serves a stale copy.

The shared conventions across all three:

```js
// works at / and under /server/, /diffuser/, /editor/
const API_ROOT = new URL('.', location.href).pathname;
function apiPath(p) { return p.startsWith('/') ? API_ROOT + p.slice(1) : p; }
```

State lives in a single `config` object, persisted to `localStorage` on
change. Option tables (sampling methods, cache types, device names) come from
`/api/status` at load, with a hardcoded fallback list so the page is usable
if the request fails.

## Editing the engine

The engine is a normal CMake project. The fastest loop is to build the target
directly and copy the artifact into the package:

```sh
cd vendor/engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target gguf-server
cp build/bin/gguf-server ../../src/ggk/server/bin/
```

Same for `diffusion-cli` → `src/ggk/diffuser/bin/diffusion` (note the target
is `diffusion-cli`, the output name is `diffusion`) and `gguf_quantizer` →
`src/ggk/editor/lib/`.

For gk itself:

```sh
cd vendor/engine/gk
cmake -B build && cmake --build build -j
./build/tests/test-foundation
./build/tests/bench-gk
```

See [engine/gk.md § Testing](engine/gk.md#testing) and
[engine/gk.md § Debug switches](engine/gk.md#debug-switches).

## Adding a control to a panel

The pattern, using a diffuser flag as the example:

1. **`engine.py`** — add the field to `DEFAULT_CONFIG` (server) or handle it
   in `build_args()`. If it is an enum, add the allow-list table and validate
   against it; unknown values should be rejected before a process is spawned.
2. **`server.py`** — expose the table in `/api/status` so the frontend can
   populate a dropdown from the engine's truth rather than a copy.
3. **`static/index.html`** — add the control, with the flag name in a
   `<code>` next to the label. Every control in the diffuser shows its flag;
   the GUI doubles as a readable CLI builder.
4. **`static/app.js`** — add it to `DEFAULT_CONFIG`, wire read/write, and
   include it in the command preview.
5. **`docs/`** — the panel's `panels.md` and `engine.md`.

Keep the allow-list in `engine.py` the authority and let the frontend's
hardcoded copy be a fallback only. That way a rebuilt engine with new options
shows them without a frontend change.

## Adding a `/api/*` route

Routes are plain `if`/`elif` chains in `do_GET` / `do_POST`. Conventions:

- Return `self._json(obj)`; errors via `self._error(msg, status)`.
- Catch `BrokenPipeError` / `ConnectionResetError` and ignore them — a
  browser that navigated away is not an error.
- `KeyError` / `ValueError` become `400`; anything else prints a traceback
  and becomes `500`.
- Long work goes in a job with an id, a status, a progress number and an
  incremental log — never a blocking request. Both the diffuser and the
  editor already do this; copy the shape.

## Testing by hand

```bash
# the CLIs
ggk --version
ggk server --help
ggk diffuser engine -- --help
ggk editor devices

# the panel APIs
curl -s localhost:8642/api/status  | python3 -m json.tool
curl -s localhost:8643/api/devices | python3 -m json.tool
curl -s localhost:8644/api/status  | python3 -m json.tool

# the GGUF parser, without the GUI
python3 -c "
from ggk.editor import gguf
p = gguf.parse_file('model.gguf')
print(p.version, len(p.metadata), len(p.tensor_infos), p.alignment)
"
```

## Version

One version for the whole suite, in `src/ggk/__init__.py`:

```python
__version__ = '0.5.1'
```

scikit-build-core reads it from there by regex
(`[tool.scikit-build.metadata.version]`), and each panel re-exports it, so
there is a single place to bump.

## Things worth knowing before you change them

- **`gui.py`'s `__class__` swap** is deliberate, not a hack looking for a
  fix. Delegating a single method would break every helper the panel handler
  calls; swapping the class makes them all resolve correctly.
- **The diffuser's progress-line collapsing** has a subtle rule: overwrite
  the previous log line only when it is a redraw of the *same* step. Collapse
  more aggressively and the polling cursor, which has already moved past the
  old entry, gets stuck showing step 1 forever.
- **The editor keeps metadata strings as raw bytes** end to end. Decoding and
  re-encoding would corrupt non-UTF-8 tokenizer entries.
- **`compute_tensor_sizes` falls back to the gap to the next tensor** for
  unknown types. That over-counts by up to `alignment - 1` bytes of padding
  but copies correctly, which is what lets an unknown type survive a
  round-trip.
- **`GGML_MAX_NAME` and `GK_MAX_NAME` are both 128** across the whole tree.
  The compat `ggml_tensor` and gk's `gk_tensor` are layout-identical by
  construction, with a static assert that fires if you widen one alone.
- **The `qz_*` codec is compiled into two artifacts.** Do not copy a block
  layout into gk; add it to the codec and let both pick it up.

## Style

Match the surrounding code. The Python is annotated, uses `from __future__
import annotations`, keeps module docstrings that explain *why* rather than
*what*, and comments the non-obvious decisions — of which there are many, and
they are load-bearing.
