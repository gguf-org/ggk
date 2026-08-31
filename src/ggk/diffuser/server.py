"""Local HTTP backend for the gguf-diffusion GUI (stdlib only).

Serves the static web GUI and a small JSON API. Model and image files are
addressed by filesystem path — the server runs on the same machine as the
browser, so nothing is ever uploaded; the GUI picks paths through the
/api/browse listing instead of drag & drop (dropped files would have to be
copied into temp storage, which multi-GB models make impractical).
"""

from __future__ import annotations

import datetime
import json
import mimetypes
import os
import pathlib
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from . import __version__, engine

STATIC_DIR = pathlib.Path(__file__).parent / "static"

MODEL_SUFFIXES = (".gguf", ".safetensors")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")

_jobs: Dict[str, engine.GenerationJob] = {}


# ── API payload helpers ──────────────────────────────────────────────────────

def _browse(path: Optional[str], kind: str) -> Dict[str, Any]:
    """List a directory for the file picker. kind filters what is shown:
    model | image | dir | any."""
    base = pathlib.Path(path).expanduser() if path else pathlib.Path.home()
    try:
        base = base.resolve()
    except OSError:
        base = pathlib.Path.home()
    if not base.is_dir():
        base = base.parent
    suffixes = {"model": MODEL_SUFFIXES, "image": IMAGE_SUFFIXES}.get(kind)
    entries = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (PermissionError, OSError):
        children = []
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            entries.append({"name": child.name, "path": str(child), "is_dir": True})
        elif kind == "dir":
            continue
        elif suffixes is None or child.suffix.lower() in suffixes:
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append({"name": child.name, "path": str(child),
                            "is_dir": False, "size": size})
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": entries}


def _output_path_for(output_dir: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(pathlib.Path(output_dir) / f"gguf-{ts}.png")


def _start_generation(body: Dict[str, Any]) -> Dict[str, Any]:
    cfg = body.get("config") or {}
    output_dir = cfg.get("output_dir") or engine.default_output_dir()
    out_dir = pathlib.Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = _output_path_for(str(out_dir))

    if cfg.get("use_custom_command"):
        args = engine.parse_custom_command(cfg.get("custom_command") or "")
    else:
        model_path = cfg.get("model_path") or ""
        if not model_path:
            raise engine.EngineError("no model file selected")
        if not os.path.isfile(model_path):
            raise engine.EngineError(f"model file not found: {model_path}")
        pack = cfg.get("tokenizer_pack_path") or ""
        if pack and not os.path.isdir(pack):
            raise engine.EngineError(f"tokenizer pack must be an existing directory: {pack}")
        args = engine.build_args(cfg, output_path)

    job = engine.GenerationJob(args, output_path)
    _jobs[job.id] = job
    return {"job_id": job.id, "output_path": output_path,
            "command": job.to_dict()["command"]}


# ── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = f"gguf-diffusion/{__version__}"

    def log_message(self, fmt, *args):  # quiet
        pass

    # -- helpers --
    def _json(self, obj: Any, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400):
        self._json({"error": message}, status)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8")) if data else {}

    def _serve_static(self, rel: str):
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._error("not found", 404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, query: str):
        params = urllib.parse.parse_qs(query)
        path = params.get("path", [""])[0]
        p = pathlib.Path(path)
        if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            self._error("not found", 404)
            return
        ctype = mimetypes.guess_type(str(p))[0] or "image/png"
        size = p.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # -- routes --
    def do_GET(self):
        path, _, query = self.path.partition("?")
        try:
            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path == "/api/status":
                self._json({
                    "version": __version__,
                    "engine_available": engine.is_available(),
                    "engine_error": engine.load_error(),
                    "engine_path": engine.binary_path(),
                    "sampling_methods": engine.SAMPLING_METHODS,
                    "schedules": engine.SCHEDULES,
                    "text_encoder_flags": engine.TEXT_ENCODER_FLAGS,
                    "additional_model_flags": engine.ADDITIONAL_MODEL_FLAGS,
                    "tensor_split_modules": engine.TENSOR_SPLIT_MODULES,
                    "tensor_split_modes": engine.TENSOR_SPLIT_MODES,
                    "home": str(pathlib.Path.home()),
                    "default_output_dir": engine.default_output_dir(),
                    "windows": os.name == "nt",
                })
            elif path == "/api/hardware":
                self._json(engine.hardware_snapshot())
            elif path == "/api/devices":
                # The engine's own device list, which is what --backend names.
                # Separate from /api/hardware because that one is nvidia-smi's
                # view and the two orderings do not agree.
                self._json({"devices": engine.devices()})
            elif path == "/api/image":
                self._serve_image(query)
            elif path.startswith("/api/job/"):
                job_id = path.rsplit("/", 1)[-1]
                params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
                job = _jobs.get(job_id)
                if job is None:
                    self._error("unknown job", 404)
                else:
                    self._json(job.to_dict(int(params.get("after", 0))))
            elif "/.." not in path and path.count("/") == 1:
                self._serve_static(path.lstrip("/"))
            else:
                self._error("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(str(e), 500)

    def do_POST(self):
        path, _, _query = self.path.partition("?")
        try:
            body = self._read_body()
            if path == "/api/browse":
                self._json(_browse(body.get("path"), body.get("kind") or "any"))
            elif path == "/api/generate":
                try:
                    self._json(_start_generation(body))
                except engine.EngineError as e:
                    self._error(str(e), 400)
            elif path.startswith("/api/job/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                job = _jobs.get(job_id)
                if job is None:
                    self._error("unknown job", 404)
                else:
                    job.cancel()
                    self._json({"ok": True})
            else:
                self._error("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError) as e:
            self._error(f"bad request: {e}", 400)
        except Exception as e:
            traceback.print_exc()
            self._error(str(e), 500)


def serve(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create the server (not yet running). port=0 picks a free port."""
    return ThreadingHTTPServer((host, port), Handler)
