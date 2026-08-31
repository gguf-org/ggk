"""Local HTTP backend for the GGUF editor GUI (stdlib only).

Serves the static web GUI and a small JSON API. Files are addressed by path —
the server runs on the same machine as the browser, so multi-GB models are
never uploaded; tensor data is streamed file-to-file on save. The one
exception is drag & drop: the browser cannot reveal a dropped file's path,
so /api/upload streams the bytes into a temp dir that is removed on exit.
"""

from __future__ import annotations

import atexit
import json
import mimetypes
import os
import pathlib
import shutil
import tempfile
import threading
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from . import __version__, gguf, quantizer
from .edits import build_edited_header, compute_final_tensors

STATIC_DIR = pathlib.Path(__file__).parent / "static"

MODEL_SUFFIXES = (".gguf", ".safetensors")

_upload_root: Optional[str] = None
_upload_lock = threading.Lock()


def _upload_dir() -> str:
    """Temp dir for drag & dropped files, cleaned up when the server exits."""
    global _upload_root
    with _upload_lock:
        if _upload_root is None:
            _upload_root = tempfile.mkdtemp(prefix="gguf-editor-drop-")
            atexit.register(shutil.rmtree, _upload_root, ignore_errors=True)
        return _upload_root


# ── jobs ─────────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "running"          # running | completed | error | cancelled
        self.stage = ""                  # human-readable current step
        self.progress = 0                # 0..100
        self.log: List[str] = []
        self.error: Optional[str] = None
        self.output_path: Optional[str] = None
        self.cancel_requested = False
        self.lock = threading.Lock()

    def log_line(self, line: str):
        with self.lock:
            self.log.append(line)

    def to_dict(self, log_after: int = 0) -> Dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "log": self.log[log_after:],
                "log_next": len(self.log),
                "error": self.error,
                "output_path": self.output_path,
            }


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _register(job: Job) -> Job:
    with _jobs_lock:
        _jobs[job.id] = job
    return job


# ── job bodies ───────────────────────────────────────────────────────────────

def _run_save(job: Job, input_path: str, output_path: str, edits: Dict[str, Any]):
    parsed = gguf.parse_file(input_path)
    finals, plans = compute_final_tensors(parsed, edits)
    header = build_edited_header(parsed, edits, finals)
    job.stage = "Writing tensor data…"

    def progress(written: int, total: int):
        job.progress = int(written * 100 / total) if total else 100
        if job.cancel_requested:
            raise quantizer.QuantizeCancelled("save cancelled")

    gguf.write_rebuilt(output_path, header, plans, parsed.alignment, progress)
    job.output_path = output_path
    job.log_line(f"saved {output_path}")


def _run_quantize(job: Job, input_path: str, output_path: str, quant: Dict[str, Any]):
    job.stage = "Quantizing…"

    def on_log(level: int, message: str):
        prefix = {quantizer.LOG_WARN: "warning: ", quantizer.LOG_ERROR: "error: "}.get(level, "")
        job.log_line(prefix + message)
        pct = quantizer.parse_progress(message)
        if pct is not None:
            job.progress = pct

    quantizer.quantize(
        input_path,
        output_path,
        default_type=quant.get("type") or None,
        tensor_type_rules=quant.get("rules") or None,
        n_threads=int(quant.get("threads") or 0),
        device=quant.get("device") or "cpu",
        log_callback=on_log,
        cancel_check=lambda: job.cancel_requested,
    )
    job.output_path = output_path


def _has_changes(edits: Dict[str, Any]) -> bool:
    return any([
        edits.get("metadata_edits"),
        edits.get("deleted_meta_keys"),
        edits.get("new_meta_rows"),
        edits.get("tensor_renames"),
        edits.get("deleted_tensors"),
        edits.get("merges"),
        edits.get("added_tensors"),
        edits.get("order"),
    ])


def _job_thread(job: Job, fn, *args):
    def run():
        try:
            fn(job, *args)
            job.status = "completed"
            job.progress = 100
        except quantizer.QuantizeCancelled:
            job.status = "cancelled"
            job.log_line("cancelled")
        except Exception as e:  # surfaced to the GUI
            job.status = "error"
            job.error = str(e)
            job.log_line(f"error: {e}")
            traceback.print_exc()
    threading.Thread(target=run, daemon=True).start()


def start_save_job(input_path: str, output_path: str, edits: Dict[str, Any]) -> Job:
    job = _register(Job("save"))
    _job_thread(job, _run_save, input_path, output_path, edits)
    return job


def start_quantize_job(input_path: str, output_path: str, quant: Dict[str, Any]) -> Job:
    job = _register(Job("quantize"))
    _job_thread(job, _run_quantize, input_path, output_path, quant)
    return job


def start_save_quantize_job(input_path: str, output_path: str,
                            edits: Dict[str, Any], quant: Dict[str, Any]) -> Job:
    """Save & Quantize: rebuild the edited file first (when there are edits),
    then run the quantizer on it — same two-step flow as the desktop editor."""
    job = _register(Job("save+quantize"))

    def body(job: Job):
        source = input_path
        temp_path = None
        try:
            if _has_changes(edits):
                fd, temp_path = tempfile.mkstemp(
                    suffix=".gguf", prefix="gguf-editor-",
                    dir=os.path.dirname(os.path.abspath(output_path)) or None)
                os.close(fd)
                job.stage = "Step 1/2: applying edits…"
                job.log_line("applying edits…")
                _run_save(job, input_path, temp_path, edits)
                source = temp_path
                job.progress = 0
            job.stage = "Step 2/2: quantizing…"
            _run_quantize(job, source, output_path, quant)
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    _job_thread(job, body)
    return job


# ── API payload helpers ──────────────────────────────────────────────────────

def _describe_file(path: str) -> Dict[str, Any]:
    parsed = gguf.parse_file(path)
    metadata = []
    for key, entry in parsed.metadata.items():
        vtype, value = entry["type"], entry["value"]
        elem_type = value["elem_type"] if vtype == gguf.ARRAY else None
        display = gguf.format_value(vtype, value)
        metadata.append({
            "key": key,
            "type": vtype,
            "type_name": (f"[{gguf.VALUE_TYPE_NAME.get(elem_type, elem_type)}]"
                          if elem_type is not None
                          else gguf.VALUE_TYPE_NAME.get(vtype, str(vtype))),
            "value": display,
            "array_len": len(value["items"]) if vtype == gguf.ARRAY else None,
        })
    tensors = [{
        "index": i,
        "name": t.name,
        "shape": t.shape,
        "dtype": t.dtype,
        "dtype_name": gguf.quantization_name(t.dtype),
        "size": parsed.tensor_sizes[i],
        "offset": t.offset,
    } for i, t in enumerate(parsed.tensor_infos)]
    return {
        "path": parsed.path,
        "file_name": os.path.basename(parsed.path),
        "file_size": parsed.file_size,
        "version": parsed.version,
        "alignment": parsed.alignment,
        "tensor_data_offset": parsed.tensor_data_offset,
        "metadata": metadata,
        "tensors": tensors,
    }


def _browse(path: Optional[str]) -> Dict[str, Any]:
    base = pathlib.Path(path).expanduser() if path else pathlib.Path.home()
    base = base.resolve()
    if not base.is_dir():
        base = base.parent
    entries = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
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
        elif child.suffix.lower() in MODEL_SUFFIXES:
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append({"name": child.name, "path": str(child),
                            "is_dir": False, "size": size})
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": entries}


# ── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = f"gguf-editor/{__version__}"
    initial_file: Optional[str] = None  # set by serve()

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
        try:
            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path == "/api/status":
                self._json({
                    "version": __version__,
                    "quantizer_available": quantizer.is_available(),
                    "quantizer_error": quantizer.load_error(),
                    "quant_types": quantizer.SUPPORTED_TYPES,
                    "initial_file": Handler.initial_file,
                    "home": str(pathlib.Path.home()),
                })
            elif path == "/api/devices":
                self._json({"devices": quantizer.list_devices()})
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

    # Raw-body route: streams a drag & dropped model file to a temp path so
    # the GUI can open it (dropped files carry no filesystem path).
    def _handle_upload(self, query: str):
        params = urllib.parse.parse_qs(query)
        name = os.path.basename(params.get("name", [""])[0]) or "dropped.gguf"
        if not name.lower().endswith(MODEL_SUFFIXES):
            self._error("Only .gguf / .safetensors files are supported", 400)
            return
        remaining = int(self.headers.get("Content-Length") or 0)
        if remaining <= 0:
            self._error("empty upload", 400)
            return
        dest = os.path.join(tempfile.mkdtemp(dir=_upload_dir()), name)
        try:
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise IOError("connection closed before the upload finished")
                    f.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            shutil.rmtree(os.path.dirname(dest), ignore_errors=True)
            raise
        self._json({"path": dest})

    def do_POST(self):
        path, _, query = self.path.partition("?")
        try:
            if path == "/api/upload":
                self._handle_upload(query)
                return
            body = self._read_body()
            if path == "/api/open":
                file_path = body.get("path", "")
                if not os.path.isfile(file_path):
                    self._error(f"File not found: {file_path}", 404)
                    return
                self._json(_describe_file(file_path))
            elif path == "/api/browse":
                self._json(_browse(body.get("path")))
            elif path == "/api/save":
                job = start_save_job(body["input_path"], body["output_path"],
                                     body.get("edits", {}))
                self._json({"job_id": job.id})
            elif path == "/api/quantize":
                job = start_quantize_job(body["input_path"], body["output_path"],
                                         body.get("quant", {}))
                self._json({"job_id": job.id})
            elif path == "/api/save-quantize":
                job = start_save_quantize_job(
                    body["input_path"], body["output_path"],
                    body.get("edits", {}), body.get("quant", {}))
                self._json({"job_id": job.id})
            elif path.startswith("/api/job/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                job = _jobs.get(job_id)
                if job is None:
                    self._error("unknown job", 404)
                else:
                    job.cancel_requested = True
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


def serve(host: str = "127.0.0.1", port: int = 0,
          initial_file: Optional[str] = None) -> ThreadingHTTPServer:
    """Create the server (not yet running). port=0 picks a free port."""
    Handler.initial_file = os.path.abspath(initial_file) if initial_file else None
    return ThreadingHTTPServer((host, port), Handler)
