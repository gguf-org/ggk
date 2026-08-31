"""Runner for the bundled gguf-server engine binary.

The engine is the C/C++ `gguf-server` from the vendored ggk engine tree
(vendor/engine, which runs on gk rather than ggml), compiled during
`pip install` (see CMakeLists.txt) and installed into this package's `bin/`
directory. One server process is managed at a time: it is spawned with its
output redirected to a log file, a watcher thread polls the TCP port until it
accepts connections, and the log tail feeds the GUI — the same flow as the
desktop app's Rust runner (gguf/src-tauri/src/llm.rs).
"""

from __future__ import annotations

import os
import pathlib
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

BIN_DIR = pathlib.Path(__file__).parent / "bin"
ENGINE_NAME = "gguf-server"
BINARY_NAME = ENGINE_NAME + ".exe" if os.name == "nt" else ENGINE_NAME

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888

# Readiness polling (llm.rs: HEALTH_POLL_MS / HEALTH_TIMEOUT_S).
HEALTH_POLL_S = 0.5
HEALTH_TIMEOUT_S = 120

# KV cache quantization types the desktop panel offers.
CACHE_TYPES = [
    "f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "bf16",
    "q6_k", "q5_k", "q4_k", "q3_k", "q2_k", "iq4_nl", "iq4_xs",
    "iq3_s", "iq3_xxs", "iq2_s", "iq2_xs", "iq2_xxs", "iq1_s",
    "tq2_0", "tq1_0", "mxfp4",
]

HOST_OPTIONS = ["127.0.0.1", "0.0.0.0", "localhost"]

# OpenAI-compatible endpoints the engine exposes (shown in the GUI).
ENDPOINTS = [
    {"method": "GET", "path": "/v1/models", "desc": "List available models"},
    {"method": "POST", "path": "/v1/chat/completions", "desc": "Chat completions (streaming supported)"},
    {"method": "POST", "path": "/v1/completions", "desc": "Text completions"},
    {"method": "POST", "path": "/v1/embeddings", "desc": "Text embeddings"},
    {"method": "POST", "path": "/v1/rerank", "desc": "Rerank documents against a query"},
    {"method": "POST", "path": "/v1/messages", "desc": "Anthropic-compatible messages"},
    {"method": "GET", "path": "/props", "desc": "Model properties and server settings"},
    {"method": "GET", "path": "/slots", "desc": "State of the inference slots"},
    {"method": "GET", "path": "/health", "desc": "Health check"},
    {"method": "GET", "path": "/metrics", "desc": "Prometheus metrics"},
]

DEFAULT_SETTINGS: Dict[str, Any] = {
    "host": DEFAULT_HOST,
    "port": DEFAULT_PORT,
    "context_length": 65536,
    "gpu_layers": 999,
    "main_gpu": 0,
    "tensor_split": "",
    "cpu_threads": 0,
    "n_parallel": 1,
    "batch_size": 2048,
    "ubatch_size": 512,
    "api_key": "",
    "model_alias": "",
    "flash_attn": True,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "cont_batching": True,
    "mlock": False,
    "no_mmap": False,
    "verbose_log": False,
}


def binary_path() -> Optional[str]:
    p = BIN_DIR / BINARY_NAME
    return str(p) if p.is_file() else None


def is_available() -> bool:
    return binary_path() is not None


def load_error() -> Optional[str]:
    if is_available():
        return None
    return (f"engine binary not found at {BIN_DIR / BINARY_NAME} — "
            "the gguf-server engine did not build during install")


class EngineError(Exception):
    pass


# ── Argument construction (mirrors llm.rs spawn_server) ──────────────────────

def _clamp(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


def build_args(cfg: Dict[str, Any]) -> List[str]:
    """Build the engine argument list from the GUI's structured config."""
    model_path = (cfg.get("model_path") or "").strip()
    if not model_path:
        raise EngineError("no model file selected")

    host = (cfg.get("host") or DEFAULT_HOST).strip() or DEFAULT_HOST
    port = _clamp(cfg.get("port", DEFAULT_PORT), 1, 65535, DEFAULT_PORT)
    ctx = _clamp(cfg.get("context_length", 4096), 512, 1048576, 4096)
    gpu_layers = _clamp(cfg.get("gpu_layers", 0), 0, 999, 0)

    args: List[str] = [
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx),
        "--n-gpu-layers", str(gpu_layers),
    ]

    if gpu_layers > 0:
        args += ["--main-gpu", str(_clamp(cfg.get("main_gpu", 0), 0, 31, 0))]
        split = (cfg.get("tensor_split") or "").strip()
        if split:
            args += ["--tensor-split", split]

    if cfg.get("verbose_log"):
        args.append("--verbose")

    mmproj = (cfg.get("mmproj_path") or "").strip()
    if mmproj:
        args += ["--mmproj", mmproj]

    threads = _clamp(cfg.get("cpu_threads", 0), 0, 4096, 0)
    if threads > 0:
        args += ["--threads", str(threads)]

    parallel = _clamp(cfg.get("n_parallel", 1), 1, 4096, 1)
    if parallel > 1:
        args += ["--parallel", str(parallel)]

    args += ["--batch-size", str(_clamp(cfg.get("batch_size", 2048), 1, 1048576, 2048))]
    args += ["--ubatch-size", str(_clamp(cfg.get("ubatch_size", 512), 1, 1048576, 512))]

    api_key = (cfg.get("api_key") or "").strip()
    if api_key:
        args += ["--api-key", api_key]

    alias = (cfg.get("model_alias") or "").strip()
    if alias:
        args += ["--alias", alias]

    args += ["--flash-attn", "on" if cfg.get("flash_attn") else "off"]

    for key, flag in (("cache_type_k", "--cache-type-k"), ("cache_type_v", "--cache-type-v")):
        val = (cfg.get(key) or "").strip()
        if val and val != "f16":
            if val not in CACHE_TYPES:
                raise EngineError(f"unknown KV cache type: {val}")
            args += [flag, val]

    if cfg.get("cont_batching", True):
        args.append("--cont-batching")
    if cfg.get("mlock"):
        args.append("--mlock")
    if cfg.get("no_mmap"):
        args.append("--no-mmap")

    mode = cfg.get("chat_template_mode") or "auto"
    if mode == "custom":
        tmpl = (cfg.get("chat_template") or "").strip()
        if tmpl:
            args += ["--chat-template", tmpl]
    elif mode == "file":
        tmpl_file = (cfg.get("chat_template_file") or "").strip()
        if tmpl_file:
            if not os.path.isfile(tmpl_file):
                raise EngineError(f"chat template file not found: {tmpl_file}")
            args += ["--chat-template-file", tmpl_file]

    return args


def parse_custom_command(command: str) -> List[str]:
    """Turn a user-edited command line into engine args.

    A leading token naming the binary ("gguf-server", "./gguf-server",
    "gguf-server.exe", or a path ending in it) is stripped — the bundled
    engine is always the one executed, same as the desktop app. The old
    "llama-server" spelling is accepted too, so saved presets keep working.
    """
    tokens = shlex.split(command, posix=(os.name != "nt"))
    if tokens:
        head = tokens[0].strip('"').replace("\\", "/").rsplit("/", 1)[-1].lower()
        if head in ("gguf-server", "gguf-server.exe",
                    "llama-server", "llama-server.exe", "server", "server.exe"):
            tokens = tokens[1:]
    if not tokens:
        raise EngineError("custom command is empty")
    return tokens


def endpoint_from_args(args: List[str]) -> Dict[str, Any]:
    """Read back host/port from an argument list (for custom commands)."""
    host, port = DEFAULT_HOST, DEFAULT_PORT
    for i, tok in enumerate(args[:-1]):
        if tok in ("--host",):
            host = args[i + 1]
        elif tok in ("--port",):
            try:
                port = int(args[i + 1])
            except ValueError:
                pass
    return {"host": host, "port": port}


# ── Process management (mirrors llm.rs) ──────────────────────────────────────

def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port_free(host: str, port: int, attempts: int = 60):
    """Give a previous server a moment to release the port (llm.rs does the same)."""
    for _ in range(attempts):
        if not _port_open(host, port, timeout=0.2):
            return
        time.sleep(0.05)


def _summarize_log(text: str) -> str:
    """First few non-empty lines, as llm.rs's summarize_output does."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    joined = " ".join(lines[-8:])
    return joined[:900] + "..." if len(joined) > 900 else joined


class ServerProcess:
    """One gguf-server engine child process.

    Output goes to a log file (the engine is chatty and long-lived, so the
    GUI tails the file by offset instead of buffering everything in memory).
    """

    def __init__(self, args: List[str], host: str, port: int, model_id: str):
        binary = binary_path()
        if binary is None:
            raise EngineError(load_error())

        self.command = [binary] + args
        self.host = host
        self.port = port
        self.model_id = model_id
        self.status = "starting"        # starting | running | stopped | error
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.ready_at: Optional[float] = None
        self.stop_requested = False
        self.lock = threading.Lock()

        fd, log_path = tempfile.mkstemp(prefix="gguf-server-", suffix=".log")
        self.log_path = log_path
        self._log_file = os.fdopen(fd, "wb")

        env = dict(os.environ)
        if "--n-gpu-layers" in args:
            idx = args.index("--n-gpu-layers")
            if idx + 1 < len(args) and args[idx + 1] == "0":
                # Hide CUDA devices when the user explicitly chose CPU-only.
                env["CUDA_VISIBLE_DEVICES"] = "-1"

        creationflags = 0
        if os.name == "nt":  # no console window pops up for the child
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.proc = subprocess.Popen(
                self.command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # so Windows finds any sibling DLLs next to the binary
                cwd=str(pathlib.Path(binary).parent),
                env=env,
                creationflags=creationflags,
            )
        except OSError as e:
            self._log_file.close()
            raise EngineError(f"failed to spawn {ENGINE_NAME}: {e}")

        threading.Thread(target=self._watch, daemon=True).start()

    # -- readiness watcher ---------------------------------------------------

    def _watch(self):
        deadline = time.time() + HEALTH_TIMEOUT_S
        while time.time() < deadline:
            with self.lock:
                if self.stop_requested:
                    return

            code = self.proc.poll()
            if code is not None:
                detail = _summarize_log(self.read_log())
                hint = ("Check: GPU Layers (try 0 for CPU-only), the model file path, "
                        "or whether the model fits in memory.")
                with self.lock:
                    if self.stop_requested:
                        self.status = "stopped"
                    else:
                        self.status = "error"
                        self.error = (
                            f"{ENGINE_NAME} exited (code {code}). {detail} {hint}".strip()
                            if code != 0 else
                            f"{ENGINE_NAME} exited before binding port {self.port}. {detail} {hint}".strip()
                        )
                return

            if _port_open(self.host, self.port):
                with self.lock:
                    if not self.stop_requested:
                        self.status = "running"
                        self.ready_at = time.time()
                return

            time.sleep(HEALTH_POLL_S)

        with self.lock:
            if not self.stop_requested and self.status == "starting":
                self.status = "error"
                self.error = (
                    f"{ENGINE_NAME} did not start within {HEALTH_TIMEOUT_S}s on "
                    f"{self.host}:{self.port}. The model may be too large for available memory."
                )

    # -- control -------------------------------------------------------------

    def stop(self):
        with self.lock:
            self.stop_requested = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        with self.lock:
            self.status = "stopped"
        try:
            self._log_file.close()
        except OSError:
            pass

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    # -- output --------------------------------------------------------------

    def read_log(self, max_bytes: int = 200_000) -> str:
        """Tail of the log file (llm.rs: read_log_tail)."""
        try:
            self._log_file.flush()
        except (OSError, ValueError):
            pass
        try:
            size = os.path.getsize(self.log_path)
            with open(self.log_path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            status, error = self.status, self.error
        if status == "running" and not self.is_alive():
            detail = _summarize_log(self.read_log())
            with self.lock:
                if not self.stop_requested:
                    self.status = status = "error"
                    self.error = error = f"{ENGINE_NAME} stopped unexpectedly. {detail}".strip()
                else:
                    self.status = status = "stopped"
        return {
            "status": status,
            "running": status == "running",
            "starting": status == "starting",
            "error": error,
            "host": self.host,
            "port": self.port,
            "url": self.url(),
            "api_base": self.url() + "/v1",
            "model_id": self.model_id,
            "uptime": round(time.time() - (self.ready_at or self.started_at), 1),
            "command": subprocess.list2cmdline(self.command),
            "log_path": self.log_path,
        }


def run_cli(argv: List[str]) -> int:
    """Run the bundled engine binary with raw arguments (CLI passthrough)."""
    binary = binary_path()
    if binary is None:
        print(f"error: {load_error()}", file=sys.stderr)
        return 1
    proc = subprocess.Popen([binary] + argv)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


# ── Hardware detection (mirrors gguf/src-tauri/src/hardware.rs) ──────────────

def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: List[str], timeout: int = 8) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _int_or_none(s: Any) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def _float_or_none(s: Any) -> Optional[float]:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _powershell_json(script: str) -> Any:
    import json
    text = _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-Command", script], timeout=20)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _cpu_info() -> Dict[str, Any]:
    info = {
        "name": platform.processor() or "Unknown CPU",
        "architecture": platform.machine() or "unknown",
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "usage_percent": None,
    }
    if os.name == "nt":
        data = _powershell_json(
            "Get-CimInstance Win32_Processor | Select-Object -First 1 "
            "Name,NumberOfCores,NumberOfLogicalProcessors,LoadPercentage | ConvertTo-Json -Compress")
        if isinstance(data, dict):
            info["name"] = (data.get("Name") or info["name"]).strip()
            info["physical_cores"] = _int_or_none(data.get("NumberOfCores"))
            info["logical_cores"] = _int_or_none(data.get("NumberOfLogicalProcessors")) or info["logical_cores"]
            info["usage_percent"] = _float_or_none(data.get("LoadPercentage"))
        return info

    if sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            info["name"] = name
        info["physical_cores"] = _int_or_none(_run(["sysctl", "-n", "hw.physicalcpu"]).strip())
        info["logical_cores"] = _int_or_none(_run(["sysctl", "-n", "hw.logicalcpu"]).strip()) or info["logical_cores"]
        return info

    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            info["name"] = line.split(":", 1)[1].strip()
            break
    core_ids = {l.split(":", 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith("core id")}
    if core_ids:
        info["physical_cores"] = len(core_ids)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    if load is not None and info["logical_cores"]:
        info["usage_percent"] = min(100.0, round(load / info["logical_cores"] * 100, 1))
    return info


def _memory_info() -> Dict[str, Any]:
    mem: Dict[str, Any] = {"total_ram_mib": None, "available_ram_mib": None}
    if os.name == "nt":
        data = _powershell_json(
            "Get-CimInstance Win32_OperatingSystem | Select-Object "
            "TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress")
        if isinstance(data, dict):
            total_kb = _int_or_none(data.get("TotalVisibleMemorySize"))
            free_kb = _int_or_none(data.get("FreePhysicalMemory"))
            mem["total_ram_mib"] = total_kb // 1024 if total_kb else None
            mem["available_ram_mib"] = free_kb // 1024 if free_kb else None
        return mem

    if sys.platform == "darwin":
        total = _int_or_none(_run(["sysctl", "-n", "hw.memsize"]).strip())
        mem["total_ram_mib"] = total // (1024 * 1024) if total else None
        return mem

    try:
        fields = {}
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key.strip()] = rest.strip()
        for src, dst in (("MemTotal", "total_ram_mib"), ("MemAvailable", "available_ram_mib")):
            if src in fields:
                mem[dst] = int(fields[src].split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return mem


def _nvidia_gpus() -> List[Dict[str, Any]]:
    smi = _which("nvidia-smi")
    if not smi:
        return []
    out = _run([smi,
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,"
                "utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        gpus.append({
            "name": parts[1],
            "vendor": "NVIDIA",
            "device_id": parts[2],
            "bus_id": parts[3],
            "vram_total_mib": _int_or_none(parts[4]),
            "vram_used_mib": _int_or_none(parts[5]),
            "utilization_percent": _float_or_none(parts[6]),
            "temperature_c": _float_or_none(parts[7]),
            "cuda": True,
        })
    return gpus


def _windows_gpus() -> List[Dict[str, Any]]:
    data = _powershell_json(
        "Get-CimInstance Win32_VideoController | Select-Object "
        "Name,AdapterRAM,PNPDeviceID,AdapterCompatibility | ConvertTo-Json -Compress")
    if isinstance(data, dict):
        data = [data]
    gpus = []
    for item in data or []:
        ram = _int_or_none(item.get("AdapterRAM"))
        gpus.append({
            "name": (item.get("Name") or "GPU").strip(),
            "vendor": (item.get("AdapterCompatibility") or "").strip() or None,
            "device_id": (item.get("PNPDeviceID") or "").strip() or None,
            "bus_id": None,
            "vram_total_mib": ram // (1024 * 1024) if ram else None,
            "vram_used_mib": None,
            "utilization_percent": None,
            "temperature_c": None,
            "cuda": False,
        })
    return gpus


def hardware_snapshot() -> Dict[str, Any]:
    """Best-effort hardware info (stdlib + nvidia-smi / PowerShell when present)."""
    gpus = _nvidia_gpus()
    if not gpus and os.name == "nt":
        gpus = _windows_gpus()

    memory = _memory_info()
    # None means "not reported"; a genuine 0 MiB reading must survive the sum
    totals = [g["vram_total_mib"] for g in gpus if g.get("vram_total_mib") is not None]
    useds = [g["vram_used_mib"] for g in gpus if g.get("vram_used_mib") is not None]
    memory["total_vram_mib"] = sum(totals) if totals else None
    memory["used_vram_mib"] = sum(useds) if useds else None

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_info(),
        "gpus": gpus,
        "memory": memory,
        "cuda_available": any(g.get("cuda") for g in gpus),
        "sampled_at_ms": int(time.time() * 1000),
    }


def recommended_settings(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Hardware-derived defaults (App.tsx: handleApplyRecommendedSettings)."""
    gpus = snapshot.get("gpus") or []
    cuda = bool(snapshot.get("cuda_available"))
    vram_total = (snapshot.get("memory") or {}).get("total_vram_mib") or 0
    low_vram = cuda and 0 < vram_total < 12288

    cuda_gpus = [(i, g) for i, g in enumerate(gpus) if g.get("cuda")]
    main_gpu, tensor_split = 0, ""
    if cuda_gpus:
        weights = [
            (i, max(1, (g.get("vram_total_mib") or 0) - (g.get("vram_used_mib") or 0)
                    if (g.get("vram_total_mib") or 0) > 0 else 1))
            for i, g in cuda_gpus
        ]
        main_gpu = max(weights, key=lambda w: w[1])[0]
        if len(weights) >= 2 and all(w > 1 for _, w in weights):
            tensor_split = ",".join(str(w) for _, w in weights)

    return {
        "context_length": 8192 if cuda else 4096,
        "gpu_layers": 999 if cuda else 0,
        "main_gpu": main_gpu,
        "tensor_split": tensor_split if cuda else "",
        "cpu_threads": 0,
        "batch_size": 2048 if cuda else 512,
        "ubatch_size": (256 if low_vram else 512) if cuda else 256,
        "flash_attn": cuda,
        "cont_batching": True,
        "mlock": False,
        "no_mmap": False,
    }
