"""Family engine registry — all families dispatched through audiocpp-fork HTTP.

No per-family Python engines are registered for inference. All inference
goes through the audiocpp_server subprocess managed by core.py.

The qwen3_tts engine class is kept on disk (engines/qwen3_tts.py) for the
Voice Studio node (AudiocoreVoiceStudio), which needs the qwen-tts Python
package for voice export/preview. It is NOT registered here — Voice Studio
imports it directly.
"""
from __future__ import annotations
from typing import Any


# No Python engines registered — all inference goes through audiocpp-fork HTTP.
_ENGINES: dict[str, str] = {}


def get_engine_class(family: str) -> Any | None:
    return None


def python_families() -> set[str]:
    return set()
