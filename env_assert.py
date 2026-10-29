"""env_assert.py — the boot-time python-env assertion (audit 016, C1).

The pack DECLARES its env at both publish doors (requirements.txt for
the Manager git-url door, pyproject [project].dependencies for the
registry door — the parity wall keeps them one set) and install.py
ASSERTS the native .so by sha256. What nothing asserted was the python
env itself. Manager re-runs requirements only when THIS pack installs
or updates; the venv is shared with every other pack and the engine
base — last installer wins. On 2026-10-23 an outside install landed
transformers 5.1.0 under qwen-tts-hf 0.1.1.post1 (floor >=5.15.1,<6):
every qwen3_tts load then died mid-run with a raw AttributeError
('Qwen3TTSTalkerCodePredictorConfig' object has no attribute
'validate_layer_type') instead of a named refusal.

The law here: the RUN-BREAKING floor gets asserted at boot, in the
pack's import path, in install.py's own refusal grammar. The exact
pin (==5.20.2) stays at the install doors — two seats, two purposes:
install converges the reproducible version, boot refuses the
run-breaking one. ComfyUI and Manager surface failed pack imports at
boot, so the user sees the named remedy in the console instead of a
crash when they press run.
"""
from __future__ import annotations

import importlib.metadata

# The floor qwen-tts-hf itself declares (importlib.metadata requires
# strings): below this, PretrainedConfig lacks validate_layer_type and
# every from_pretrained dies at config construction.
TRANSFORMERS_FLOOR = "5.15.1"
TRANSFORMERS_CEILING = "6"

REMEDY = (
    "melite-audio-nodes: the shared python env holds transformers {have} "
    "but the qwen3-tts lane requires >={floor},<{ceil}. Another pack or "
    "the engine base likely re-pinned it after this pack installed. Remedy: "
    "pip install 'transformers>={floor},<{ceil}' (or ComfyUI-Manager → "
    "Update this pack, which re-runs its requirements)."
)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def assert_transformers_floor() -> None:
    """Refuse the pack import unless transformers meets the qwen floor.

    @raises RuntimeError with the named remedy when the env un-converged.
    """
    have = "?"
    try:
        have = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            REMEDY.format(have="nothing (not installed)", floor=TRANSFORMERS_FLOOR, ceil=TRANSFORMERS_CEILING)
        ) from None
    if not (_version_tuple(have) >= _version_tuple(TRANSFORMERS_FLOOR) and _version_tuple(have) < _version_tuple(TRANSFORMERS_CEILING)):
        raise RuntimeError(
            REMEDY.format(have=have, floor=TRANSFORMERS_FLOOR, ceil=TRANSFORMERS_CEILING)
        )


CUDA_REMEDY = (
    "melite-audio-nodes: no CUDA device is visible to this python env "
    "(torch.cuda.is_available() is False). The audio lanes have no CPU "
    "mode worth serving: the moss_sfx_v2 pipeline silently falls back to "
    "CPU at generate time (pipeline_moss_soundeffect.py), turning every "
    "run into a multi-minute crawl. Remedy: start the engine on a host "
    "with the GPU visible (check nvidia-smi and CUDA_VISIBLE_DEVICES), "
    "or stop loading this pack on CPU-only hosts."
)


def assert_cuda_lane() -> None:
    """Refuse the pack import when the GPU lane is absent.

    The 2026-10-29 roar burn's "it ran on CPU" half: nothing asserted
    the device at boot, so a CUDA-less env served a silent CPU
    fallback instead of a named refusal.

    @raises RuntimeError with the named remedy when no CUDA device.
    """
    try:
        import torch
    except ImportError:
        return  # the transformers floor already names a broken torch-less env
    if not torch.cuda.is_available():
        raise RuntimeError(CUDA_REMEDY)
