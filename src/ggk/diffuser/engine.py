"""Runner for the bundled diffusion engine binary.

The engine is a single CLI executable compiled during `pip install` (see
CMakeLists.txt) and installed into this package's `bin/` directory. Each
generation spawns one engine process; a reader thread captures its output,
parses the progress bars, and collects the produced image files — the same
flow as the desktop app's Rust runner (gguf/src-tauri/src/diffusion.rs).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

BIN_DIR = pathlib.Path(__file__).parent / "bin"
BINARY_NAME = "diffusion.exe" if os.name == "nt" else "diffusion"

# Name tables mirror the engine (diffusion/src/gk-diffuser.cpp); the
# desktop panel exposes the same options.
SAMPLING_METHODS = [
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep",
    "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge",
]

SCHEDULES = [
    "discrete", "karras", "exponential", "ays", "gits", "sgm_uniform",
    "simple", "smoothstep", "kl_optimal", "lcm", "bong_tangent", "ltx2",
    "logit_normal", "flux2", "flux", "beta",
]

TEXT_ENCODER_FLAGS = [
    "--llm", "--clip_l", "--clip_g", "--clip_vision", "--t5xxl", "--llm_vision",
    "--llm-adapter",
]

ADDITIONAL_MODEL_FLAGS = [
    "--high-noise-diffusion-model", "--uncond-diffusion-model", "--control-net",
    "--photo-maker", "--upscale-model", "--taesd", "--embeddings-connectors",
    "--audio-vae",
]

# Modules --backend/--split-mode can address (diffusion/src/core/ggml_extend_backend.h).
TENSOR_SPLIT_MODULES = [
    "diffusion", "te", "clip-vision", "vae", "control-net", "photo-maker", "upscaler",
]
TENSOR_SPLIT_MODES = ["layer", "row"]

# "  |====>     | 5/20 - 1.23it/s" (progress bars are \r-separated)
PROGRESS_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*-")


def binary_path() -> Optional[str]:
    p = BIN_DIR / BINARY_NAME
    return str(p) if p.is_file() else None


def is_available() -> bool:
    return binary_path() is not None


def load_error() -> Optional[str]:
    if is_available():
        return None
    return (f"engine binary not found at {BIN_DIR / BINARY_NAME} — "
            "the diffusion engine did not build during install")


class EngineError(Exception):
    pass


def _push(args: List[str], flag: str, value: Any):
    args.append(flag)
    args.append(str(value))


def build_args(cfg: Dict[str, Any], output_path: str) -> List[str]:
    """Build the engine argument list from the GUI's structured config.

    Mirrors the desktop app's build_generation_args, plus the image-input
    flags the desktop only exposed through its command preview.
    """
    args: List[str] = []

    model_path = cfg.get("model_path") or ""
    if model_path:
        flag = "--model" if cfg.get("model_mode") == "model" else "--diffusion-model"
        _push(args, flag, model_path)

    if cfg.get("vae_path"):
        _push(args, "--vae", cfg["vae_path"])
    if cfg.get("tokenizer_pack_path"):
        _push(args, "--tokenizer-pack", cfg["tokenizer_pack_path"])

    for m in cfg.get("additional_models") or []:
        if m.get("path") and m.get("flag") in ADDITIONAL_MODEL_FLAGS:
            _push(args, m["flag"], m["path"])

    for enc in cfg.get("text_encoders") or []:
        if enc.get("path") and enc.get("flag") in TEXT_ENCODER_FLAGS:
            _push(args, enc["flag"], enc["path"])

    if cfg.get("init_image_path"):
        _push(args, "--init-img", cfg["init_image_path"])
        _push(args, "--strength", f"{float(cfg.get('strength', 0.75)):.2f}")
    if cfg.get("mask_image_path"):
        _push(args, "--mask", cfg["mask_image_path"])
    if cfg.get("end_image_path"):
        _push(args, "--end-img", cfg["end_image_path"])
    if cfg.get("control_image_path"):
        _push(args, "--control-image", cfg["control_image_path"])
        _push(args, "--control-strength", f"{float(cfg.get('control_strength', 0.9)):.2f}")

    ref_images = [r for r in (cfg.get("ref_image_paths") or []) if r]
    for ref in ref_images:
        _push(args, "--ref-image", ref)
    if ref_images:
        if cfg.get("increase_ref_index"):
            args.append("--increase-ref-index")
        if cfg.get("disable_auto_resize_ref_image"):
            args.append("--disable-auto-resize-ref-image")

    if cfg.get("prompt"):
        _push(args, "-p", cfg["prompt"])
    if cfg.get("negative_prompt"):
        _push(args, "-n", cfg["negative_prompt"])

    _push(args, "--cfg-scale", f"{float(cfg.get('cfg_scale', 1.0)):.2f}")
    _push(args, "--steps", int(cfg.get("steps", 4)))
    _push(args, "-W", min(2048, max(64, int(cfg.get("width", 512)))))
    _push(args, "-H", min(2048, max(64, int(cfg.get("height", 512)))))

    seed = int(cfg.get("seed", -1))
    if seed >= 0:
        _push(args, "--seed", seed)

    method = cfg.get("sampling_method") or ""
    if method:
        if method not in SAMPLING_METHODS:
            raise EngineError(f"unknown sampling method: {method}")
        _push(args, "--sampling-method", method)

    schedule = cfg.get("schedule") or ""
    if schedule and schedule != "discrete":
        if schedule not in SCHEDULES:
            raise EngineError(f"unknown schedule: {schedule}")
        _push(args, "--scheduler", schedule)

    if int(cfg.get("batch_count", 1)) > 1:
        _push(args, "--batch-count", int(cfg["batch_count"]))

    if cfg.get("flash_attn"):
        args.append("--diffusion-fa")
    if cfg.get("mmap"):
        args.append("--mmap")
    if cfg.get("offload_to_cpu"):
        args.append("--offload-to-cpu")
    if cfg.get("clip_on_cpu"):
        args.append("--clip-on-cpu")
    if cfg.get("vae_on_cpu"):
        args.append("--vae-on-cpu")
    if cfg.get("control_net_cpu"):
        args.append("--control-net-cpu")

    split_rows = [r for r in (cfg.get("tensor_split") or [])
                  if r.get("module") and r.get("devices")]
    if split_rows:
        backend_val = ",".join(
            f"{r['module']}={'&'.join(r['devices'])}" for r in split_rows
        )
        _push(args, "--backend", backend_val)
        row_mode = [r for r in split_rows if r.get("split_mode") == "row"]
        if row_mode:
            split_mode_val = ",".join(f"{r['module']}=row" for r in row_mode)
            _push(args, "--split-mode", split_mode_val)

    if cfg.get("verbose"):
        args.append("-v")

    _push(args, "-o", output_path)
    return args


def parse_custom_command(command: str) -> List[str]:
    """Turn a user-edited command line into engine args.

    A leading token naming the binary ("diffusion", "./diffusion",
    "diffusion.exe", or a path ending in it) is stripped — the bundled engine
    is always the one executed, same as the desktop app.
    """
    tokens = shlex.split(command, posix=(os.name != "nt"))
    if tokens:
        head = tokens[0].strip('"').replace("\\", "/").rsplit("/", 1)[-1].lower()
        if head in ("diffusion", "diffusion.exe", "diffusion-cli", "diffusion-cli.exe"):
            tokens = tokens[1:]
    if not tokens:
        raise EngineError("custom command is empty")
    return tokens


def find_output_files(expected_path: str) -> List[str]:
    """Find output image files: `base.png` for single images, or
    `base_0000.png`… for batches (same probing as the desktop app)."""
    path = pathlib.Path(expected_path)
    if path.is_file():
        return [str(path)]
    stem, ext = path.stem, path.suffix
    files = sorted(
        str(p) for p in path.parent.glob(f"{stem}*{ext}") if p.is_file()
    ) if path.parent.is_dir() else []
    return files


def _summarize_log(lines: List[str]) -> str:
    """Last error-looking line, else the last non-empty line."""
    for line in reversed(lines):
        low = line.lower()
        if "error" in low or "failed" in low or "invalid" in low:
            return line.strip()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""


class GenerationJob:
    """One engine process. Log lines, progress, and results are read by the
    HTTP server; all mutable state is guarded by `lock`."""

    def __init__(self, args: List[str], expected_output: str, cwd: Optional[str] = None):
        binary = binary_path()
        if binary is None:
            raise EngineError(load_error())

        self.id = uuid.uuid4().hex[:12]
        self.command = [binary] + args
        self.expected_output = expected_output
        self.status = "running"        # running | completed | error | cancelled
        self.log: List[str] = []
        self.step = 0
        self.steps = 0
        self.error: Optional[str] = None
        self.output_files: List[str] = []
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.cancel_requested = False
        self.lock = threading.Lock()

        creationflags = 0
        if os.name == "nt":  # allow killing without a console window popping up
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=cwd or str(pathlib.Path(expected_output).parent),
            creationflags=creationflags,
            bufsize=0,  # unbuffered pipe so \r progress bars arrive live
        )
        threading.Thread(target=self._reader, daemon=True).start()

    # -- output capture ------------------------------------------------------

    def _append_line(self, line: str):
        line = line.rstrip()
        if not line:
            return
        # strip ANSI erase sequences left by the progress printer
        line = line.replace("\033[K", "")
        m = PROGRESS_RE.search(line)
        with self.lock:
            if m:
                self.step, self.steps = int(m.group(1)), int(m.group(2))
                # \r-refreshed bars would flood the log: overwrite the
                # previous line, but only when it's a redraw of the *same*
                # step (updated s/it estimate) -- once the step number
                # advances, keep it as a new entry so the log_after polling
                # cursor (which has already moved past the old entry) can
                # still deliver it to the client instead of the line getting
                # stuck showing the first step forever.
                prev = self.log[-1] if self.log else None
                prev_m = PROGRESS_RE.search(prev) if prev else None
                if prev_m and prev_m.group(1) == m.group(1):
                    self.log[-1] = line
                    return
            self.log.append(line)

    def _reader(self):
        buf = b""
        stream = self.proc.stdout
        while True:
            chunk = stream.read(4096) if stream else b""
            if not chunk:
                break
            buf += chunk
            # progress bars end in \r, normal lines in \n
            while True:
                cut = min((i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0),
                          default=-1)
                if cut < 0:
                    break
                self._append_line(buf[:cut].decode("utf-8", "replace"))
                buf = buf[cut + 1:]
        if buf:
            self._append_line(buf.decode("utf-8", "replace"))

        code = self.proc.wait()
        outputs = find_output_files(self.expected_output)
        with self.lock:
            self.finished_at = time.time()
            if self.cancel_requested:
                self.status = "cancelled"
            elif outputs:
                self.status = "completed"
                self.output_files = outputs
                self.step = self.steps or self.step
            else:
                self.status = "error"
                if code == 0:
                    hint = "Process finished but no output file was found."
                else:
                    hint = f"diffusion exited with an error (exit code {code})."
                detail = _summarize_log(self.log)
                self.error = f"{hint} {detail}".strip()

    # -- control -------------------------------------------------------------

    def cancel(self):
        with self.lock:
            self.cancel_requested = True
        if self.proc.poll() is None:
            self.proc.terminate()

            def _hard_kill():
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            threading.Thread(target=_hard_kill, daemon=True).start()

    def to_dict(self, log_after: int = 0) -> Dict[str, Any]:
        with self.lock:
            progress = int(self.step * 100 / self.steps) if self.steps else 0
            return {
                "id": self.id,
                "status": self.status,
                "step": self.step,
                "steps": self.steps,
                "progress": progress,
                "log": self.log[log_after:],
                "log_next": len(self.log),
                "error": self.error,
                "output_files": self.output_files,
                "elapsed": round((self.finished_at or time.time()) - self.started_at, 1),
                "command": subprocess.list2cmdline(self.command),
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


def default_output_dir() -> str:
    downloads = pathlib.Path.home() / "Downloads"
    return str(downloads if downloads.is_dir() else pathlib.Path.home())


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def hardware_snapshot() -> Dict[str, Any]:
    """Best-effort hardware info (stdlib + nvidia-smi when present)."""
    info: Dict[str, Any] = {
        "os": sys.platform,
        "cpu_count": os.cpu_count(),
        "total_ram_mib": None,
        "available_ram_mib": None,
        "gpus": [],
    }
    try:
        meminfo = pathlib.Path("/proc/meminfo").read_text()
        fields = {}
        for line in meminfo.splitlines():
            key, _, rest = line.partition(":")
            fields[key.strip()] = rest.strip()
        for src, dst in (("MemTotal", "total_ram_mib"), ("MemAvailable", "available_ram_mib")):
            if src in fields:
                info[dst] = int(fields[src].split()[0]) // 1024
    except OSError:
        pass
    smi = _which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    info["gpus"].append({
                        "name": parts[0],
                        "vram_total_mib": _int_or_none(parts[1]),
                        "vram_used_mib": _int_or_none(parts[2]),
                        "utilization_percent": _int_or_none(parts[3]),
                        "temperature_c": _int_or_none(parts[4]),
                    })
        except (OSError, subprocess.TimeoutExpired):
            pass
    return info


def _int_or_none(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None


_devices_cache: Optional[List[Dict[str, Any]]] = None


def devices(refresh: bool = False) -> List[Dict[str, Any]]:
    """The compute devices the engine can address, asked of the engine itself.

    The names here are the only ones ``--backend`` accepts. They cannot be
    derived from nvidia-smi: CUDA numbers its devices fastest-first by default,
    so on a machine with an RTX 5090 and an RTX 4050 the engine's ``cuda0`` is
    the 5090 while nvidia-smi calls the 4050 GPU 0. Guessing gets the split
    backwards, silently.

    A device the build has no kernels for is not listed — the engine drops it
    at discovery rather than failing partway through a generation.
    """
    global _devices_cache
    if _devices_cache is not None and not refresh:
        return _devices_cache

    result: List[Dict[str, Any]] = []
    binary = binary_path()
    if binary:
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = subprocess.run(
                [binary, "--list-devices"],
                capture_output=True, text=True, timeout=30,
                creationflags=creationflags,
            )
            if out.returncode == 0:
                parsed = json.loads(out.stdout)
                if isinstance(parsed, list):
                    result = [d for d in parsed if isinstance(d, dict) and d.get("name")]
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            # An engine too old for the flag, or one that could not start. The
            # GUI falls back to a free-text device box, which still works.
            pass

    _devices_cache = result
    return result
