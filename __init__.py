"""audiocpp-fork — ComfyUI custom nodes for the audiocpp-fork C++ audio engine.

All inference runs NATIVELY IN-PROCESS via libaudiocore_native.so (ctypes).
No HTTP server, no subprocess — the engine_runtime shared library is loaded
directly into the Python process, exactly like every other ComfyUI model.

Families: moss_tts, qwen3_tts, ace_step, moss_sfx_v2

The .qvoice Voice Studio features (export/preview) use the qwen-tts Python
package directly — they are development/authoring tools, not inference.

Model folders are DECLARED, never hardcoded: the pack resolves them via
ComfyUI's folder_paths, and plugins/melite-manager/manager/extra_model_paths.yaml
is the single source of truth (`audiocpp` → engine GGUFs root, `qwen3_tts` →
Voice Studio Python-engine HF source dirs). The env vars below are ONLY
the standalone/test escape hatch OUTSIDE ComfyUI (no defaults — a folder
that is neither declared nor overridden fails loud at import):

    AUDIOCORE_NATIVE_LIB    — path to libaudiocore_native.so
    AUDIOCPP_MODEL_SPECS_DIR — path to model_specs/ directory
    AUDIOCPP_MODELS_DIR     — engine GGUFs root (standalone)
    QWEN3_HF_ROOT           — Voice Studio HF source dirs root (standalone)
"""
from __future__ import annotations

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# No WEB_DIRECTORY: this pack ships no JS extensions (the ./web dir the old
# declaration pointed at never existed).

logger = logging.getLogger("audiocore-nodes")


def _patch_cleanup_models_gc():
    """Patch cleanup_models_gc to also evict stale LoadedModel entries."""
    try:
        from comfy import model_management
        _original = model_management.cleanup_models_gc

        def _patched():
            _original()
            try:
                model_management.cleanup_models()
            except Exception:
                pass

        model_management.cleanup_models_gc = _patched
    except (ImportError, AttributeError):
        pass


_patch_cleanup_models_gc()
