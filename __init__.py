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
ComfyUI's folder_paths, and the endpoint statement's extra_model_folders
is the single source of truth (`audiocpp` → engine GGUFs root,
`qwen3_tts` → Voice Studio Python-engine HF source dirs). VENDOR
RE-CURE (2026-11-01) — COMFYUI EXTENSION NEVER USE ENV VARIABLES: the
old standalone/test env seats (AUDIOCORE_NATIVE_LIB,
AUDIOCPP_MODEL_SPECS_DIR, AUDIOCPP_MODELS_DIR, QWEN3_HF_ROOT) are
retired. An operator-built .so lands in the pack's native/ directory;
model specs resolve by the model-path-anchored walk; tests stub the
folder_paths module. A folder that is not seated fails loud at import,
naming the category seat.
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


