"""Local HTTP backend for the gguf-server GUI (stdlib only).

Serves the static web GUI and a small JSON API that starts, stops and watches
one gguf-server engine process. Model and template files are addressed by
filesystem path — this backend runs on the same machine as the browser, so
nothing is ever uploaded; the GUI picks paths through the /api/browse listing
instead of drag & drop (dropped files would have to be copied into temp
storage, which multi-GB models make impractical).
"""

from __future__ import annotations

import atexit
import json
import mimetypes
import os
import pathlib
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import __version__, engine

STATIC_DIR = pathlib.Path(__file__).parent / "static"

MODEL_SUFFIXES = (".gguf",)
TEMPLATE_SUFFIXES = (".json", ".jinja", ".jinja2", ".txt")

# One managed server at a time, exactly like the desktop app.
_server: Optional[engine.ServerProcess] = None
_server_lock = threading.Lock()


def _stop_current():
    global _server
    with _server_lock:
        current, _server = _server, None
    if current is not None:
        current.stop()


atexit.register(_stop_current)


# ── API payload helpers ──────────────────────────────────────────────────────

def _browse(path: Optional[str], kind: str) -> Dict[str, Any]:
    """List a directory for the file picker. kind filters what is shown:
    model | template | dir | any."""
    base = pathlib.Path(path).expanduser() if path else pathlib.Path.home()
    try:
        base = base.resolve()
    except OSError:
        base = pathlib.Path.home()
    if not base.is_dir():
        base = base.parent
    suffixes = {"model": MODEL_SUFFIXES, "template": TEMPLATE_SUFFIXES}.get(kind)
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


def _start_server(body: Dict[str, Any]) -> Dict[str, Any]:
    global _server

    cfg = body.get("config") or {}

    if cfg.get("use_custom_command"):
        args = engine.parse_custom_command(cfg.get("custom_command") or "")
        endpoint = engine.endpoint_from_args(args)
        host, port = endpoint["host"], endpoint["port"]
        model_id = cfg.get("model_alias") or "model"
    else:
        model_path = (cfg.get("model_path") or "").strip()
        if not model_path:
            raise engine.EngineError(
                "No model file selected. Go to the Model tab and choose a .gguf file.")
        if not os.path.isfile(model_path):
            raise engine.EngineError(f"Model file not found: {model_path}")
        mmproj = (cfg.get("mmproj_path") or "").strip()
        if mmproj and not os.path.isfile(mmproj):
            raise engine.EngineError(f"mmproj file not found: {mmproj}")
        args = engine.build_args(cfg)
        host = (cfg.get("host") or engine.DEFAULT_HOST).strip() or engine.DEFAULT_HOST
        port = int(cfg.get("port") or engine.DEFAULT_PORT)
        model_id = (cfg.get("model_alias") or "").strip() or \
            pathlib.Path(model_path).stem

    _stop_current()
    engine.wait_for_port_free(host, port)

    proc = engine.ServerProcess(args, host, port, model_id)
    with _server_lock:
        _server = proc
    return proc.to_dict()


def _server_state() -> Dict[str, Any]:
    with _server_lock:
        current = _server
    if current is None:
        return {"status": "stopped", "running": False, "starting": False,
                "error": None, "url": "", "api_base": "", "log_path": None}
    return current.to_dict()


# ── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = f"gguf-server/{__version__}"

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

    # -- routes --
    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query)
        try:
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
            elif path == "/api/status":
                self._json({
                    "version": __version__,
                    "engine_available": engine.is_available(),
                    "engine_error": engine.load_error(),
                    "engine_path": engine.binary_path(),
                    "engine_name": engine.BINARY_NAME,
                    "cache_types": engine.CACHE_TYPES,
                    "host_options": engine.HOST_OPTIONS,
                    "endpoints": engine.ENDPOINTS,
                    "defaults": engine.DEFAULT_SETTINGS,
                    "home": str(pathlib.Path.home()),
                    "windows": os.name == "nt",
                })
            elif path == "/api/hardware":
                self._json(engine.hardware_snapshot())
            elif path == "/api/recommended":
                snapshot = engine.hardware_snapshot()
                self._json({"hardware": snapshot,
                            "settings": engine.recommended_settings(snapshot)})
            elif path == "/api/server":
                self._json(_server_state())
            elif path == "/api/log":
                with _server_lock:
                    current = _server
                max_bytes = int(params.get("max_bytes", ["200000"])[0])
                self._json({"log": current.read_log(max_bytes) if current else ""})
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
            elif path == "/api/start":
                try:
                    self._json(_start_server(body))
                except engine.EngineError as e:
                    self._error(str(e), 400)
            elif path == "/api/stop":
                _stop_current()
                self._json(_server_state())
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
    """Create the GUI server (not yet running). port=0 picks a free port."""
    return ThreadingHTTPServer((host, port), Handler)


def shutdown_engine():
    """Stop the managed engine process, if any (used on GUI shutdown)."""
    _stop_current()
