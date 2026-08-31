"""ctypes bindings for the bundled gguf_quantizer shared library.

The library is the standalone (ggml-free) quantizer from the engine tree, built
by this package's CMakeLists.txt and installed into ggk/editor/lib
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import re
import sys
import threading
from typing import Callable, List, Optional

# gqz_log_level
LOG_INFO, LOG_WARN, LOG_ERROR = 0, 1, 2

_LOG_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_CANCEL_CB_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class _GqzParams(ctypes.Structure):
    _fields_ = [
        ("input_path", ctypes.c_char_p),
        ("output_path", ctypes.c_char_p),
        ("default_type", ctypes.c_char_p),
        ("tensor_type_rules", ctypes.c_char_p),
        ("n_threads", ctypes.c_int),
        ("device", ctypes.c_char_p),
        ("log_cb", _LOG_CB_TYPE),
        ("log_user_data", ctypes.c_void_p),
        ("cancel_cb", _CANCEL_CB_TYPE),
        ("cancel_user_data", ctypes.c_void_p),
    ]


def _lib_paths() -> List[pathlib.Path]:
    base = pathlib.Path(__file__).parent / "lib"
    if sys.platform.startswith("win") or sys.platform == "cygwin":
        names = ["gguf_quantizer.dll", "libgguf_quantizer.dll"]
    elif sys.platform == "darwin":
        names = ["libgguf_quantizer.dylib", "libgguf_quantizer.so"]
    else:
        names = ["libgguf_quantizer.so"]
    return [base / n for n in names]


_lib = None
_lib_error: Optional[str] = None
_lib_lock = threading.Lock()


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _lib_error
    with _lib_lock:
        if _lib is not None or _lib_error is not None:
            return _lib
        for path in _lib_paths():
            if not path.exists():
                continue
            try:
                if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(path.parent))
                lib = ctypes.CDLL(str(path))
            except OSError as e:
                # Python 3.8+ resolves DLL dependencies without consulting
                # PATH; retry with the legacy search so a MinGW-built DLL can
                # still find runtime DLLs (e.g. libwinpthread-1.dll) there.
                lib = None
                if sys.platform.startswith("win"):
                    try:
                        lib = ctypes.CDLL(str(path), winmode=0)
                    except OSError:
                        lib = None
                if lib is None:
                    _lib_error = f"Failed to load {path}: {e}"
                    return None
            lib.gqz_default_params.restype = _GqzParams
            lib.gqz_default_params.argtypes = []
            lib.gqz_get_device_count.restype = ctypes.c_int
            lib.gqz_get_device_count.argtypes = []
            lib.gqz_get_device_name.restype = ctypes.c_char_p
            lib.gqz_get_device_name.argtypes = [ctypes.c_int]
            lib.gqz_get_device_description.restype = ctypes.c_char_p
            lib.gqz_get_device_description.argtypes = [ctypes.c_int]
            lib.gqz_quantize.restype = ctypes.c_int
            lib.gqz_quantize.argtypes = [ctypes.POINTER(_GqzParams)]
            _lib = lib
            return _lib
        _lib_error = (
            "gguf_quantizer shared library not found in "
            f"{pathlib.Path(__file__).parent / 'lib'} — reinstall the package "
            "(pip install ggk) so the C/C++ engine gets built."
        )
        return None


def is_available() -> bool:
    return _load() is not None


def load_error() -> Optional[str]:
    _load()
    return _lib_error


# Supported --type values, kept in sync with the quantizer CLI's list.
SUPPORTED_TYPES = [
    "f32", "f16", "q4_0", "q4_1", "q5_0",
    "q5_1", "q8_0", "q8_1", "q2_k",
    "q3_k", "q4_k", "q5_k", "q6_k",
    "q8_k", "iq2_xxs", "iq2_xs", "iq3_xxs",
    "iq1_s", "iq4_nl", "iq3_s", "iq2_s",
    "iq4_xs", "i8", "i16", "i32", "i64",
    "f64", "iq1_m", "bf16", "tq1_0", "tq2_0",
    "mxfp4", "nvfp4", "q1_0", "q2_0",
]


def list_devices() -> List[dict]:
    """Devices compiled into this build; index 0 is always \"cpu\"."""
    lib = _load()
    if lib is None:
        return []
    out = []
    for i in range(lib.gqz_get_device_count()):
        out.append({
            "name": lib.gqz_get_device_name(i).decode("utf-8", "replace"),
            "description": lib.gqz_get_device_description(i).decode("utf-8", "replace"),
        })
    return out


_PROGRESS_RE = re.compile(r"^\[\s*(\d+)%\]\s*")


def parse_progress(message: str) -> Optional[int]:
    """Extract the percentage from a '[ 42%] tensor ...' log line."""
    m = _PROGRESS_RE.match(message)
    return int(m.group(1)) if m else None


class QuantizeError(RuntimeError):
    pass


class QuantizeCancelled(QuantizeError):
    pass


def quantize(
    input_path: str,
    output_path: str,
    default_type: Optional[str] = None,
    tensor_type_rules: Optional[str] = None,
    n_threads: int = 0,
    device: str = "cpu",
    log_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Run the quantizer. Raises QuantizeError on failure, QuantizeCancelled
    when cancel_check returned True (the partial output file is removed by
    the library)."""
    lib = _load()
    if lib is None:
        raise QuantizeError(load_error() or "quantizer library unavailable")
    if not default_type and not tensor_type_rules:
        raise QuantizeError("provide a default type and/or tensor type rules")

    params = lib.gqz_default_params()
    params.input_path = os.fspath(input_path).encode("utf-8")
    params.output_path = os.fspath(output_path).encode("utf-8")
    params.default_type = default_type.encode("utf-8") if default_type else None
    params.tensor_type_rules = (
        tensor_type_rules.encode("utf-8") if tensor_type_rules else None
    )
    params.n_threads = int(n_threads)
    params.device = (device or "cpu").encode("utf-8")

    last_error: List[str] = []

    def _log(level, message, _user):
        text = (message or b"").decode("utf-8", "replace")
        if level == LOG_ERROR:
            last_error.append(text)
        if log_callback:
            log_callback(level, text)

    def _cancel(_user):
        return 1 if (cancel_check and cancel_check()) else 0

    # keep callback objects referenced for the duration of the C call
    log_cb = _LOG_CB_TYPE(_log)
    cancel_cb = _CANCEL_CB_TYPE(_cancel)
    params.log_cb = log_cb
    params.cancel_cb = cancel_cb

    rc = lib.gqz_quantize(ctypes.byref(params))
    if rc == 2:
        raise QuantizeCancelled("quantization cancelled")
    if rc != 0:
        raise QuantizeError(last_error[-1] if last_error else f"quantizer failed (exit code {rc})")
