"""melite-audio-nodes — the audio-core ComfyUI pack (all families).

Inference seats per family (the engines/ docstring carries the full
law): the C++-engine families load in-process through the native
ctypes binding, moss_sfx_v2 loads a pure torch pipeline
(engines/moss_sfx_v2.py), and the qwen3_tts voice-design path rides
the qwen-tts Python package. Engine-GC hygiene (stale LoadedModel
eviction) is applied by core.py at import — shared law, not this
pack's private side effect.

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

# THE BOOT ENV ASSERT (audit 016, 2026-11-12): declare-only is not a
# contract. The venv is shared and Manager re-runs requirements only on
# this pack's install/update — an outside pin un-converges an installed
# machine silently and every qwen3_tts run dies with a raw
# AttributeError. Refuse the IMPORT instead, with the named remedy
# (ComfyUI/Manager surface failed pack imports at boot). Same grammar
# as install.py's sha256 gate and the allocator preflight.
from .env_assert import assert_cuda_lane, assert_transformers_floor

assert_transformers_floor()
# The 2026-10-29 roar burn's "it ran on CPU" half: the moss pipeline
# falls back to CPU silently at generate time — refuse at boot instead.
assert_cuda_lane()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# No WEB_DIRECTORY: this pack ships no JS extensions (the ./web dir the old
# declaration pointed at never existed).


