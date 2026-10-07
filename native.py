"""ctypes binding for libaudiocore_native.so — the in-process audiocpp engine.

ELIMINATES THE HTTP SERVER. This module loads the C++ engine_runtime directly
into the Python process via ctypes. No subprocess, no HTTP, no port management.
The model persists in GPU memory between generations, exactly like every other
ComfyUI model (SDXL, Flux, etc.).

The C ABI is defined in app/native_api.h in the audiocpp-fork source tree.
Every function that can fail returns a sentinel (NULL or success=0) and stores
a thread-local error retrievable via last_error().
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("audiocore-nodes")

# ─── Library discovery ────────────────────────────────────────────────────

# The estate's managed runtime also lands the asset (its provision
# sync fetches the same pinned release into
# <melite>/data/runtime/audiocore/, a sibling of the engine tree at
# <melite>/data/runtime/comfyui), so from custom_nodes/<pack>/ that
# landing is three parents up, one sibling over. Packs installed
# outside that layout have no sibling — None, not a crash. The pack's
# own native/ landing (install.py's fetch) is checked before this.
_ESTATE_RUNTIME = (
    Path(__file__).resolve().parents[3] / "audiocore"
    if len(Path(__file__).resolve().parents) > 3 else None
)


def _find_native_lib() -> str:
    """Resolve the path to libaudiocore_native.so."""
    # 1. Explicit env var — an operator-built .so overrides everything
    env_path = os.environ.get("AUDIOCORE_NATIVE_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. The pack's own install.py landing — Manager runs it on
    #    install; it fetches + sha-verifies the pinned release asset
    #    into this pack's native/ directory. A plain ComfyUI with
    #    this pack Manager-installed converges here, estate or not.
    pack_local = Path(__file__).resolve().parent / "native" / "libaudiocore_native.so"
    if pack_local.is_file():
        return str(pack_local)

    # 3. The estate provision landing + a conventional host install
    candidates = [
        *([] if _ESTATE_RUNTIME is None
           else [_ESTATE_RUNTIME / "libaudiocore_native.so"]),
        Path("/usr/local/lib/libaudiocore_native.so"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    raise RuntimeError(
        "libaudiocore_native.so not found. The pack's install.py fetches\n"
        "it (re-run the ComfyUI-Manager install), or set\n"
        "AUDIOCORE_NATIVE_LIB to a .so you built.\n"
        f"Checked: env AUDIOCORE_NATIVE_LIB, {pack_local}, {[str(c) for c in candidates]}"
    )


# ─── C struct definitions ─────────────────────────────────────────────────


class _AudioOutput(ctypes.Structure):
    """Matches audiocore_audio_t in native_api.h."""
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("num_samples", ctypes.c_int64),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("success", ctypes.c_int),
    ]


# Progress callback: void(int64_t step, int64_t total, const char* label)
_PROGRESS_CB = ctypes.CFUNCTYPE(
    None, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p,
)


# ─── Library loader ───────────────────────────────────────────────────────


def _load_lib() -> ctypes.CDLL:
    lib_path = _find_native_lib()
    logger.info("loading libaudiocore_native from %s", lib_path)
    lib = ctypes.CDLL(lib_path)

    # ── Signatures ──
    lib.audiocore_last_error.restype = ctypes.c_char_p
    lib.audiocore_last_error.argtypes = []

    lib.audiocore_registry_create.restype = ctypes.c_void_p
    lib.audiocore_registry_create.argtypes = []

    lib.audiocore_registry_destroy.restype = None
    lib.audiocore_registry_destroy.argtypes = [ctypes.c_void_p]

    lib.audiocore_session_create.restype = ctypes.c_void_p
    lib.audiocore_session_create.argtypes = [
        ctypes.c_void_p,   # registry
        ctypes.c_char_p,   # family
        ctypes.c_char_p,   # model_path
        ctypes.c_char_p,   # task
        ctypes.c_char_p,   # backend
        ctypes.c_int,      # device
        ctypes.c_int,      # threads
        ctypes.c_char_p,   # model_spec_override
        ctypes.c_char_p,   # variant
        ctypes.c_char_p,   # load_options_json
    ]

    lib.audiocore_session_destroy.restype = None
    lib.audiocore_session_destroy.argtypes = [ctypes.c_void_p]

    lib.audiocore_session_run.restype = _AudioOutput
    lib.audiocore_session_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    lib.audiocore_audio_free.restype = None
    lib.audiocore_audio_free.argtypes = [ctypes.POINTER(_AudioOutput)]

    lib.audiocore_session_set_progress.restype = None
    lib.audiocore_session_set_progress.argtypes = [
        ctypes.c_void_p, _PROGRESS_CB,
    ]

    lib.audiocore_session_family.restype = ctypes.c_char_p
    lib.audiocore_session_family.argtypes = [ctypes.c_void_p]

    lib.audiocore_session_vram_used_bytes.restype = ctypes.c_int64
    lib.audiocore_session_vram_used_bytes.argtypes = [ctypes.c_void_p]

    return lib


# Module-level singleton — loaded once, reused for all sessions
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


class NativeError(Exception):
    """Raised when the native library reports an error."""


def _check_error() -> None:
    """Raise if the library set a thread-local error."""
    lib = _get_lib()
    err = lib.audiocore_last_error()
    if err:
        raise NativeError(err.decode("utf-8"))


# ─── Public Python API ────────────────────────────────────────────────────


class NativeRegistry:
    """Wraps the C++ ModelRegistry — creates loaders for all families."""

    def __init__(self) -> None:
        lib = _get_lib()
        self._handle = lib.audiocore_registry_create()
        if not self._handle:
            _check_error()
            raise NativeError("registry_create returned NULL with no error message")

    def destroy(self) -> None:
        if self._handle:
            _get_lib().audiocore_registry_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


# Module-level singleton registry — created once, reused for all sessions
_registry: Optional[NativeRegistry] = None


def get_registry() -> NativeRegistry:
    global _registry
    if _registry is None:
        _registry = NativeRegistry()
        logger.info("native registry created")
    return _registry


class NativeSession:
    """Wraps a loaded model session (audiocore_session_t).

    The model is loaded into GPU memory on construction and stays resident
    until destroy() is called. Multiple runs reuse the same loaded weights.
    """

    def __init__(
        self,
        registry: NativeRegistry,
        family: str,
        model_path: str,
        task: str = "tts",
        backend: str = "cuda",
        device: int = 0,
        threads: int = 4,
        model_spec_override: Optional[str] = None,
        variant: Optional[str] = None,
        load_options: Optional[dict[str, Any]] = None,
    ) -> None:
        lib = _get_lib()
        self._handle = None
        self._progress_cb_ref = None  # keep the ctypes callback alive

        load_json = json.dumps(load_options) if load_options else None

        self._handle = lib.audiocore_session_create(
            registry._handle,
            family.encode("utf-8"),
            model_path.encode("utf-8"),
            task.encode("utf-8"),
            backend.encode("utf-8"),
            device,
            threads,
            (model_spec_override or "").encode("utf-8"),
            (variant or "").encode("utf-8"),
            (load_json or "").encode("utf-8") if load_json else None,
        )
        if not self._handle:
            _check_error()
            raise NativeError(
                f"session_create returned NULL for family={family} path={model_path}"
            )
        logger.info(
            "native session created: family=%s path=%s task=%s backend=%s",
            family, model_path, task, backend,
        )

    def run(self, request: dict[str, Any]) -> tuple[list[float], int, int]:
        """Run inference. Returns (pcm_samples, sample_rate, channels).

        The request dict matches the server's /v1/audio/speech body format.
        All params flow through to the engine — no filtering, no drops.
        """
        lib = _get_lib()
        request_json = json.dumps(request)

        audio = lib.audiocore_session_run(
            self._handle, request_json.encode("utf-8"),
        )
        try:
            if not audio.success:
                _check_error()
                raise NativeError("session_run returned failure with no error message")

            # Copy the C float* to a Python list, then free the C buffer
            n = audio.num_samples
            if n > 0:
                samples = [audio.samples[i] for i in range(n)]
            else:
                samples = []

            return samples, audio.sample_rate, audio.channels
        finally:
            lib.audiocore_audio_free(ctypes.byref(audio))

    def set_progress(
        self, cb: Optional[Callable[[int, int, str], None]]
    ) -> None:
        """Install a progress callback. cb(step, total, label)."""
        lib = _get_lib()
        if cb is None:
            self._progress_cb_ref = None
            lib.audiocore_session_set_progress(self._handle, _PROGRESS_CB(0))
            return

        def _wrapper(step: int, total: int, label_ptr) -> None:
            try:
                label = label_ptr.decode("utf-8") if label_ptr else ""
                cb(step, total, label)
            except Exception:
                pass  # progress is best-effort

        self._progress_cb_ref = _PROGRESS_CB(_wrapper)
        lib.audiocore_session_set_progress(self._handle, self._progress_cb_ref)

    @property
    def family(self) -> str:
        lib = _get_lib()
        raw = lib.audiocore_session_family(self._handle)
        return raw.decode("utf-8") if raw else ""

    def vram_used(self) -> int:
        lib = _get_lib()
        return lib.audiocore_session_vram_used_bytes(self._handle)

    def destroy(self) -> None:
        if self._handle:
            _get_lib().audiocore_session_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
