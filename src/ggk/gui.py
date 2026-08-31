"""Unified ggk GUI: one HTTP server, three panels.

The server, diffuser and editor panels keep their own backends (handler
classes, engine runners, static frontends) exactly as in their standalone
packages; this module mounts them under /server/, /diffuser/ and /editor/ on
a single port and serves the shell page that switches between them.

Dispatch works by rewriting the request path to strip the panel prefix and
handing the request to the panel's own BaseHTTPRequestHandler subclass (a
temporary __class__ swap, restored afterwards, so every helper method the
panel handler calls resolves on the right class). The panel frontends build
API URLs relative to the page, so the same files work mounted at a prefix
here and at the root when a panel runs standalone.
"""

from __future__ import annotations

import mimetypes
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .server import server as server_backend
from .diffuser import server as diffuser_backend
from .editor import server as editor_backend

STATIC_DIR = pathlib.Path(__file__).parent / "static"

PANELS = {
    "server": server_backend.Handler,
    "diffuser": diffuser_backend.Handler,
    "editor": editor_backend.Handler,
}


class Handler(BaseHTTPRequestHandler):
    server_version = f"ggk/{__version__}"

    def log_message(self, fmt, *args):  # quiet
        pass

    # -- panel dispatch --
    def _dispatch(self, method: str):
        clean = self.path.partition("?")[0]
        segment = clean.split("/", 2)[1] if clean.startswith("/") else ""
        panel_cls = PANELS.get(segment)
        if panel_cls is not None:
            prefix = "/" + segment
            if clean == prefix:
                # /server -> /server/ so the panel's relative URLs resolve
                self.send_response(307)
                self.send_header("Location", prefix + "/" + self.path[len(prefix):].lstrip("/"))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.path = self.path[len(prefix):]
            shell_cls = self.__class__
            try:
                self.__class__ = panel_cls
                getattr(self, method)()
            finally:
                self.__class__ = shell_cls
            return
        if method == "do_GET":
            if clean in ("/", "/index.html"):
                self._serve_static("index.html")
                return
            if "/.." not in clean and clean.count("/") == 1:
                self._serve_static(clean.lstrip("/"))
                return
        self._not_found()

    def do_GET(self):
        try:
            self._dispatch("do_GET")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self._dispatch("do_POST")
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- shell statics --
    def _not_found(self):
        body = b'{"error": "not found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel: str):
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._not_found()
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create the unified GUI server (not yet running). port=0 picks a free port."""
    return ThreadingHTTPServer((host, port), Handler)


def shutdown_engines():
    """Stop anything the panels left running (used on GUI shutdown)."""
    server_backend.shutdown_engine()
