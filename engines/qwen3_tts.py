"""qwen3_tts — text-to-speech via the faster-qwen3-tts package.

Wraps ``FasterQwen3TTS`` (pip-installed faster-qwen3-tts, torch
backend), which itself wraps the official ``qwen_tts``
Qwen3TTSModel. No GGUF, no conversion. The model is loaded directly from the HF source dir
(config.json + model.safetensors + speech_tokenizer/ + tokenizer bits).

Three call modes match the audiocore surface:
  - tts         -> generate_custom_voice(text, speaker, language)
  - voice_clone -> generate_voice_clone(text, ref_audio, ref_text)
                   (needs a "base" model variant, not "custom_voice")
  - design      -> generate_voice_design(text, instruct)
                   (needs a "voicedesign" model variant)

Hyperparameters (temperature, top_p, top_k, repetition_penalty) are
forwarded to Faster's generate_* calls. ``seed`` is applied via
torch.manual_seed only — Faster's generate_* take no seed kwarg, and
the old qwen **kwargs path forwarded it dead into generate() anyway.
``emotion`` is likewise NOT forwarded (no Faster/qwen param carries
it); the wdelta preview path logs a warning when it is dropped.

Voice Studio extensions (export / preview / load_qvoice) implement the
.qvoice file format (lite + WDELTA) and the combo pipeline. See
``qvoice.py`` for the file format spec.
"""
from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from typing import Any

from ..core import resolve_model_folder

logger = logging.getLogger("audiocore.qwen3_tts")

# The voices-bank neutral cache (the F51 identity steering): bank dir +
# dim → (newest .qvoice mtime, mean embedding | None). Refreshed when the
# bank's file set changes; entries are small (one vector per bank).
_BANK_NEUTRAL_CACHE: dict[tuple[str, int], tuple[float, Any]] = {}


def _tolerant_default_rope_parameters(config, device=None, seq_len=None,
                                      layer_type=None):
    """RoPE init tolerant of configs without ``rope_theta`` (transformers 5.20).

    qwen-tts-hf registers its own ``"default"`` RoPE initializer that reads
    ``config.rope_theta`` unconditionally. Custom Qwen3-TTS configs carry it,
    but the speech tokenizer's stock ``MimiConfig`` (transformers ≥5.x) keeps
    RoPE params in ``rope_parameters`` instead — so the first weight load
    crashes in ``initialize_weights`` with
    ``AttributeError: 'MimiConfig' object has no attribute 'rope_theta'``.

    THIS INIT IS **NOT** THROWAWAY FOR MIMI (engine audit 2026-11-14, D1 —
    the old docstring claimed qwen's ``from_pretrained`` rebuilds every
    ``inv_freq`` via ``restore_rope_buffers``; false for the exact model
    this protects): ``restore_rope_buffers`` only rebuilds modules
    carrying BOTH ``rope_init_fn`` AND ``inv_freq``, and transformers'
    ``MimiRotaryEmbedding`` keeps ``rope_init_fn`` as a local variable —
    so the codec's rope is NEVER restored, and this function's output IS
    the final Mimi rope state at decode time (``inv_freq`` is a
    non-persistent buffer, absent from the safetensors). The REAL
    invariant: the fallback math below must equal transformers' own
    ``MimiRotaryEmbedding.compute_default_rope_parameters`` — same
    ``rope_parameters["rope_theta"]`` base, same
    ``head_dim * partial_rotary_factor`` dim, ``attention_factor`` 1.0 —
    and :func:`_assert_mimi_rope` refuses the load when it does not.
    This wrapper delegates to qwen's exact math for normal configs and
    falls back to ``rope_parameters["rope_theta"]`` (else 10000.0, the
    transformers default theta) only for configs that lack
    ``rope_theta`` — same shape, same buffers, load proceeds to the
    codec decode.
    """
    from qwen_tts import _transformers_compat as _qt_compat
    try:
        return _qt_compat._default_rope_parameters(
            config, device, seq_len, layer_type)
    except AttributeError:
        pass
    import torch
    base = 10000.0
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict):
        base = rp.get("rope_theta", 10000.0) or 10000.0
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


@contextlib.contextmanager
def _rope_registry_patched():
    """The load-scoped RoPE patch (engine audit 2026-11-14, D1; the
    20:56 cold-boot burn, D5).

    Installs the tolerant ``"default"`` initializer around ONE
    ``from_pretrained`` call, then restores whatever entry was live
    before (qwen_tts's own, since its import already re-seated the
    registry) — the override never leaks past the load. The entry is
    read WITHOUT assuming it exists: a cold interpreter may carry no
    ``"default"`` seat yet (the D5 ``KeyError('default')`` fired at
    exactly this read), so absent restores to absent — nesting and
    repeated loads stay safe warm and cold.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    saved = ROPE_INIT_FUNCTIONS.get("default")
    ROPE_INIT_FUNCTIONS["default"] = _tolerant_default_rope_parameters
    try:
        yield
    finally:
        if saved is None:
            ROPE_INIT_FUNCTIONS.pop("default", None)
        else:
            ROPE_INIT_FUNCTIONS["default"] = saved


def _assert_mimi_rope(handle: Any) -> None:
    """Refuse LOUD when the codec's loaded rope diverges from the
    transformers math (engine audit 2026-11-14, D1 arm 4).

    The tolerant initializer's output IS the final Mimi rope state:
    ``restore_rope_buffers`` never touches the codec (transformers'
    ``MimiRotaryEmbedding`` keeps ``rope_init_fn`` as a local variable,
    not ``self.rope_init_fn``), and ``inv_freq`` is a non-persistent
    buffer absent from the safetensors — so nothing overwrites it after
    the load. The invariant this asserts: every loaded codec rope
    module's ``inv_freq`` equals transformers' own
    ``MimiRotaryEmbedding.compute_default_rope_parameters`` on that
    module's config. A checkpoint whose true theta differs from the
    fallback, or a future transformers math change, turns from silent
    texture corruption into this refusal. Zero rope modules found is
    also loud — that is drift, not health.
    """
    import torch
    from transformers.models.mimi import modeling_mimi
    tokenizer = getattr(handle, "speech_tokenizer", None)
    if tokenizer is None:
        return  # no codec on this handle (non-audio variant) — nothing to pin
    rope_modules = [
        m for m in tokenizer.modules()
        if isinstance(m, modeling_mimi.MimiRotaryEmbedding)
    ]
    if not rope_modules:
        raise RuntimeError(
            "qwen3_tts: the speech tokenizer exposes no "
            "MimiRotaryEmbedding modules — the rope invariant cannot be "
            "pinned (transformers drift?). Refusing rather than "
            "asserting nothing."
        )
    for module in rope_modules:
        expected, _ = (
            modeling_mimi.MimiRotaryEmbedding
            .compute_default_rope_parameters(module.config)
        )
        got = module.inv_freq.to(device=expected.device,
                                 dtype=expected.dtype)
        if not torch.allclose(got, expected.to(device=got.device), atol=1e-6):
            raise RuntimeError(
                "qwen3_tts: codec rope divergence at load — the loaded "
                "inv_freq does not equal transformers' "
                "MimiRotaryEmbedding math for this config (theta or dim "
                "mismatch between the checkpoint and the tolerant "
                "fallback). Refusing rather than decoding on wrong rope."
            )
    logger.debug("qwen3_tts: codec rope pinned (%d modules)",
                 len(rope_modules))


# THE STATIC-CACHE BUDGET (ported from ComfyUI-Qwen3-TTS-Fast's loader,
# nodes.py: its max_seq_len widget defaults 2048 and every generate
# path clamps max_new_tokens to max_seq_len - 256 — beyond the static
# cache the captured graphs cannot hold the decode, so the budget,
# not faster's 2048 generate default, owns the ceiling).
_MAX_SEQ_LEN = 2048
_MAX_NEW_TOKENS = _MAX_SEQ_LEN - 256
# THE LOAD HEADROOM (the third Triton burn, 2026-10-22: the studio
# eviction yields torch-managed VRAM, but the foreign block — the
# prompt's own ggml native session plus leftovers no eviction owns —
# stays resident. The capture's side-stream allocations then starve
# and the warmup Triton matmul dies with "(cpu tensor?)", aborting
# the process AFTER the weights already loaded. Gate the load LOUD
# below this floor instead of burning the process: the weights'
# safetensors bytes plus graphs + capture workspace + margin.
_LOAD_HEADROOM_BYTES = 4_000_000_000


def _preinit_static_cache(static_cache: Any, config: Any,
                          dtype: Any, device: Any) -> None:
    """Allocate the StaticCache layer buffers ahead of graph capture.

    Ported from ComfyUI-Qwen3-TTS-Fast nodes.py
    (``_preinit_static_cache``): faster-qwen3-tts calls
    ``layer.lazy_initialization(k, v)`` (the transformers 5.x
    signature); 4.x takes a single key tensor and derives value from
    its shape — identical result either way, since the library passes
    the same tensor twice. Pre-initing here sets ``is_initialized``,
    so the library's own loop becomes a no-op on either version.
    """
    num_kv = (getattr(config, "num_key_value_heads", None)
              or config.num_attention_heads)
    head_dim = (getattr(config, "head_dim", None)
                or (config.hidden_size // config.num_attention_heads))
    import torch
    dummy = torch.zeros(1, num_kv, 1, head_dim, dtype=dtype, device=device)
    for layer in static_cache.layers:
        if getattr(layer, "is_initialized", False):
            continue
        try:
            layer.lazy_initialization(dummy, dummy)  # transformers 5.x
        except TypeError:
            layer.lazy_initialization(dummy)  # transformers 4.x


def _apply_compat(handle: Any) -> None:
    """Pre-init both graphs' static caches (Fast loader spelling).

    Runs between from_pretrained and warmup so the library's own
    cache-init loop is a no-op and capture meets initialized layers
    on transformers 4.x AND 5.x.
    """
    talker = handle.talker_graph
    _preinit_static_cache(talker.static_cache, talker.model.config,
                          talker.dtype, talker.device)
    predictor = handle.predictor_graph
    _preinit_static_cache(predictor.static_cache, predictor.pred_model.config,
                          predictor.dtype, predictor.device)


def _safetensors_bytes(hf_dir: str) -> int:
    """Sum the *.safetensors weights in the HF dir (the load's cost)."""
    total = 0
    try:
        for name in os.listdir(hf_dir):
            if name.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(hf_dir, name))
    except OSError:
        pass
    return total


def _require_load_headroom(hf_dir: str, variant: str) -> None:
    """Refuse LOUD when the card cannot fit the load (never Triton-die).

    The burn's lesson: eager warmup passes and the capture-stream
    run dies with "(cpu tensor?)" — not a CPU tensor (the same
    tensors pass eagerly and standalone), a starved allocator
    handing the Triton kernel a bad pointer AFTER the weights
    already loaded. Driver-level free (torch's view AND the foreign
    native block) must cover the weights + graphs + workspace, or
    the press refuses naming the numbers and the remedy (free the
    card — the Unload node now also releases the warmed torch copy
    — and press again).
    """
    need = _safetensors_bytes(hf_dir) + _LOAD_HEADROOM_BYTES
    if need <= _LOAD_HEADROOM_BYTES:
        raise RuntimeError(
            f"qwen3_tts: no safetensors weights under {hf_dir} "
            f"(variant={variant}) — refusing the load, not weighing "
            f"an empty dir."
        )
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
    except Exception as e:
        logger.warning("qwen3_tts: cannot query VRAM free (%s); "
                       "proceeding — the load owns the verdict", e)
        return
    if free_b < need:
        raise RuntimeError(
            f"qwen3_tts: card cannot fit variant={variant} — "
            f"driver free {free_b / 1e9:.1f} GB < "
            f"need {need / 1e9:.1f} GB "
            f"(weights {_safetensors_bytes(hf_dir) / 1e9:.1f} GB + "
            f"graphs/workspace {_LOAD_HEADROOM_BYTES / 1e9:.1f} GB). "
            f"The capture starves below this floor and dies as "
            f"'Pointer argument cannot be accessed from Triton'. "
            f"Free the card and press again: the estate's free door "
            f"(comfy_manage free) chains /audiocore/release_warmed "
            f"and drops these warmed torch seats too; the Unload "
            f"Audiocore Model node releases them as well."
        )
    logger.info("qwen3_tts: load headroom ok (free %.1f GB >= need %.1f GB)",
                free_b / 1e9, need / 1e9)


def release_warmed_handles() -> int:
    """Release every warmed torch handle (the Fast Unload semantics).

    The explicit yield gesture: drops the process-wide cache so the
    ~4.3 GB + graphs return to the card. The next voice run reloads
    and re-captures (slow once), which is the documented trade the
    Unload node offers. Returns the released count for the log.
    """
    n = len(_WARMED_HANDLES)
    _WARMED_HANDLES.clear()
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    logger.info("qwen3_tts: released %d warmed handle(s)", n)
    return n


# Process-wide warmed torch handles, keyed by (resolved HF dir, variant).
# UPSTREAM'S LAW (ported 2026-10-22 from the Faster-Qwen3TTS ComfyUI
# extension: its loader caches one model per key in ``_models`` and
# every consumer node generates through the cached handle): each
# studio run used to construct a fresh engine that reloaded ~4.3 GB
# and re-captured the CUDA graphs, while the loader ALSO held the
# same weights as a native session. After a heavy ComfyUI run (a
# ~20 GB staged model on a 24 GB card), the repeat load + graph
# capture starved VRAM and the warmup's Triton matmul died with
# "Pointer argument cannot be accessed from Triton (cpu tensor?)",
# then the process itself aborted (c10::AcceleratorError). One copy
# per key, warmed once at load, shared by every run in the process.
_WARMED_HANDLES: dict[tuple[str, str], Any] = {}


def _load_warmed_handle(hf_dir: str, variant: str) -> Any:
    """Return the warmed FasterQwen3TTS handle for (hf_dir, variant).

    Cache hit: the already-loaded, already-warmed handle (no reload,
    no graph re-capture). Cache miss: load from the HF dir, capture
    the CUDA graphs NOW (upstream's ``warmup=True`` loader default —
    never lazily mid-run under pressure), then cache. The single
    loading spelling: ``load()`` and ``_ensure_variant`` both ride
    this seat.
    """
    key = (hf_dir, variant)
    cached = _WARMED_HANDLES.get(key)
    if cached is not None:
        logger.info("qwen3_tts: reusing warmed variant=%s from %s",
                    variant, hf_dir)
        return cached
    import contextlib
    import io
    import torch
    # Silence stdout on import (flash-attn warning) so it doesn't
    # pollute ComfyUI logs.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise RuntimeError(
                "qwen3_tts: faster-qwen3-tts Python package not "
                f"installed: {e}"
            ) from e
    logger.info("qwen3_tts: loading variant=%s from %s", variant, hf_dir)
    # CUDA graph capture (warmup below) requires the native caching
    # allocator. ComfyUI defaults to cudaMallocAsync for torch 2.0+
    # (--cuda-malloc); under it the capture-time Triton matmul dies
    # with "Pointer argument cannot be accessed from Triton (cpu
    # tensor?)" even with the card 90% free (fourth burn — bisected
    # standalone: async allocator burns, native passes). Refuse loud
    # with the cure named rather than dying opaquely mid-capture.
    backend = torch.cuda.get_allocator_backend()
    if backend == "cudaMallocAsync":
        raise RuntimeError(
            "qwen3_tts: CUDA allocator backend is 'cudaMallocAsync' — "
            "CUDA graph capture is incompatible with it. Restart "
            "ComfyUI with --disable-cuda-malloc (native caching "
            "allocator) to load the qwen3-tts variants."
        )
    # Gate the allocation BEFORE the weights land: below this floor
    # the capture starves and Triton-dies after loading (third burn).
    _require_load_headroom(hf_dir, variant)
    # THE LOAD-SCOPED ROPE PATCH (engine audit 2026-11-14, D1): the
    # tolerant "default" initializer is installed around THIS
    # from_pretrained only, then qwen's own registry entry is restored —
    # the process-global registry never keeps melite's override, so
    # other transformers families loaded in this interpreter keep
    # qwen's semantics, not ours.
    with _rope_registry_patched():
        handle = FasterQwen3TTS.from_pretrained(
            hf_dir,
            device="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            max_seq_len=_MAX_SEQ_LEN,
        )
    # The codec rope invariant: the tolerant fallback's output is the
    # FINAL Mimi rope state (nothing restores it) — pin it to
    # transformers' own math or refuse the load.
    _assert_mimi_rope(handle)
    # THE SHIM LOAD PATH IS LOAD-BEARING (engine audit 2026-11-14 §2):
    # the tokenizer rides qwen's Qwen3TTSTokenizer.from_pretrained
    # (restore_rope_buffers + restore_mimi_full_attention), which forces
    # Mimi to decode FULL causal (window ≥ max_position_embeddings).
    # Loading the tokenizer by ANY other path leaves the checkpoint's
    # declared sliding_window 250/72 enforced at forward — decode
    # texture degrades EVERYWHERE, silently. Never add a second
    # tokenizer-load spelling.
    # Pre-init the static caches (Fast loader spelling) so the
    # library's own init loop is a no-op on transformers 4.x/5.x.
    _apply_compat(handle)
    # Capture at LOAD, not lazily inside the first generate: a cold
    # capture racing a full card is the allocation that died.
    handle.warmup()
    _WARMED_HANDLES[key] = handle
    logger.info("qwen3_tts: variant=%s warmed and cached (device=%s)",
                variant, handle.device)
    return handle


# Default hyperparameters — match the audiocore node defaults where possible.
# speaker defaults to "" (unstated): _custom_voice LOUD-refuses an empty
# speaker on the plain-TTS path — the silent vivian fallback is dead
# (QVOICE ARCHITECTURE LAWS). The nodes.py face maps voice="none" → ""
# and strips "" upstream; the wdelta combo path below passes an explicit
# speaker="vivian" (stated, not a default).
_DEFAULTS = {
    "speaker": "",
    "language": "english",
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
}


class Qwen3TtsEngine:
    """Wraps faster_qwen3_tts.FasterQwen3TTS to match audiocore's session contract."""

    def __init__(self) -> None:
        self.model: Any = None          # faster_qwen3_tts.FasterQwen3TTS
        self._model_dir: str | None = None
        # Variant currently loaded: "customvoice" | "base" | "voicedesign".
        # Tracked so Voice Studio preview can decide whether the active
        # model is the right variant for the qvoice kind being previewed
        # (lite → base, wdelta → customvoice-after-patch).
        self._variant: str = "customvoice"

    # ── audiocore session contract ──────────────────────────────────────

    @property
    def _qwen(self) -> Any:
        """The wrapped qwen_tts Qwen3TTSModel inside FasterQwen3TTS.

        QWEN_TTS-ONLY handle: used solely for the two APIs Faster does
        not surface (``get_supported_speakers``,
        ``create_voice_clone_prompt``) and for locating the talker
        submodule in the WDELTA patch protocol. All synthesis rides
        ``self.model`` (Faster). Raises while unloaded.
        """
        if self.model is None:
            raise RuntimeError("qwen3_tts: engine not loaded")
        return self.model.model

    def load(self, path: str, **extras: Any) -> bool:
        """Load FasterQwen3TTS from `path` (HF source dir with config.json).

        Variant selection: features gate per model variant. The default
        load picks CustomVoice (plain TTS with predefined speakers).
        Pass `variant='base'` via extras for voice cloning, or
        `variant='voicedesign'` for instructed voice design.

        extras (JSON object from the LoadAudiocoreModel node):
          - variant:           'customvoice' (default) | 'base' | 'voicedesign'
          - speaker_encoder_path: ignored (the Base HF dir ships its own)
        """
        variant_hint = str(extras.get("variant", "")).lower().strip()
        # Treat empty/missing hint as None so _resolve_model_dir uses the
        # path-suffix detection (default → customvoice).
        hint = variant_hint or None
        hf_dir = self._resolve_model_dir(path, variant_hint=hint)
        if hf_dir is None:
            raise RuntimeError(
                f"qwen3_tts: could not resolve an HF source dir from {path!r} "
                f"(variant_hint={hint!r}). Point model_path at a Qwen3-TTS "
                f"HF dir (containing config.json + model.safetensors)."
            )
        # Track which variant was resolved so preview/export know what's live.
        self._variant = self._detect_variant_from_dir(hf_dir)

        self.model = _load_warmed_handle(hf_dir, self._variant)
        self._model_dir = hf_dir
        logger.info(
            "qwen3_tts: model ready (device=%s, speakers=%s)",
            self.model.device,
            (self._qwen.get_supported_speakers() or [])[:5],
        )
        return True

    def run_tts(self, text: str, **kwargs: Any) -> tuple[list[float], int]:
        """Generate speech.

        Dispatches based on `mode`:
          - "voice_clone" (or any mode + reference_audio)  → clone path
          - "design" + instruct                            → design path
          - otherwise                                      → custom_voice
        """
        if self.model is None:
            raise RuntimeError("qwen3_tts: engine not loaded")

        # Pop audiocore control kwargs that don't map to generate_*.
        mode = str(kwargs.get("mode", "tts")).lower()
        # `voice` is audiocore's speaker alias. Unstated speaker stays ""
        # — _custom_voice refuses it LOUD (no silent preset fallback).
        speaker = str(kwargs.get("voice", kwargs.get("speaker", _DEFAULTS["speaker"])))
        language = self._coerce_language(kwargs.get("language", _DEFAULTS["language"]))
        instruct = kwargs.get("instruct", "") or ""
        seed = self._coerce_seed(kwargs.get("seed", 0))

        # Build gen kwargs forwarded to Faster's generate_*.
        gen_kwargs: dict[str, Any] = {}
        for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if k in kwargs and kwargs[k] is not None and kwargs[k] != "":
                try:
                    gen_kwargs[k] = float(kwargs[k]) if k != "top_k" else int(kwargs[k])
                except (TypeError, ValueError):
                    pass
        # The captured graphs hold _MAX_SEQ_LEN positions — clamp the
        # decode length to the static-cache budget (Fast _clamp_tokens
        # spelling), never faster's 2048 generate default.
        gen_kwargs["max_new_tokens"] = self._clamp_max_new_tokens(
            kwargs.get("max_new_tokens", _MAX_NEW_TOKENS))
        # seed conditions torch RNG only — Faster's generate_* take no
        # seed kwarg, so it is NEVER forwarded (would TypeError).
        if seed > 0:
            import torch
            torch.manual_seed(seed)

        # Dispatch.
        ref_audio = kwargs.get("reference_audio", "") or ""
        ref_text = kwargs.get("reference_text", "") or ""
        if mode in ("voice_clone", "clone") or ref_audio:
            return self._voice_clone(text, language, ref_audio, ref_text,
                                     kwargs, gen_kwargs)
        if mode in ("design", "voice_design") and instruct:
            return self._voice_design(text, language, instruct, gen_kwargs)

        # Default: plain TTS via custom_voice.
        return self._custom_voice(text, speaker, language, instruct, gen_kwargs)

    def run_music(self, caption: str, **kwargs: Any) -> tuple[list[float], int, int]:
        raise RuntimeError("qwen3_tts does not support music generation")

    def compute_embedding(self, wav_path: str) -> list[float]:
        """Compute a speaker embedding from a WAV via the loaded model.

        Uses the wrapped qwen handle's create_voice_clone_prompt to
        extract the x-vector embedding from the reference audio. Returns
        it as a flat list so the AudiocoreVoiceEmbedding node can cache
        + reuse it.
        """
        if self.model is None:
            raise RuntimeError("qwen3_tts: engine not loaded")
        if not wav_path or not os.path.isfile(wav_path):
            raise RuntimeError(
                f"qwen3_tts: compute_embedding needs a valid wav_path, "
                f"got {wav_path!r}")

        # create_voice_clone_prompt returns a LIST of VoiceClonePromptItem
        # (one per reference sample). Take the first.
        # x_vector_only_mode=True skips the in-context-learning path so we
        # don't need a ref_text transcript — we only want the x-vector.
        prompt_items = self._qwen.create_voice_clone_prompt(
            ref_audio=wav_path,
            ref_text=None,
            x_vector_only_mode=True,
        )
        if not prompt_items:
            raise RuntimeError(
                "qwen3_tts: create_voice_clone_prompt returned an empty list"
            )
        prompt_item = prompt_items[0]
        # Pull the speaker embedding out of the VoiceClonePromptItem
        # dataclass (qwen_tts names it `ref_spk_embedding`).
        emb = None
        if hasattr(prompt_item, "ref_spk_embedding"):
            emb = getattr(prompt_item, "ref_spk_embedding", None)
        elif hasattr(prompt_item, "x_vector"):
            emb = getattr(prompt_item, "x_vector", None)
        elif isinstance(prompt_item, dict):
            emb = (prompt_item.get("ref_spk_embedding")
                   or prompt_item.get("x_vector")
                   or prompt_item.get("embedding"))
        if emb is None:
            raise RuntimeError(
                "qwen3_tts: could not extract x-vector from voice-clone "
                "prompt; the loaded model may not support speaker-encoder "
                "embedding extraction."
            )
        import torch
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().float().view(-1).tolist()
        import numpy as np
        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32).reshape(-1).tolist()
        return list(emb)

    def unload(self) -> None:
        if self.model is not None:
            # Drop the underlying nn.Module + processor references.
            del self.model
            self.model = None
            self._variant = "customvoice"
        # Release the process-wide warmed cache too (Fast Unload
        # semantics): an explicit unload returns the ~4.3 GB + graphs
        # to the card. _drop_model keeps the cache (variant switches
        # stay fast); unload() is the yield gesture and owns the
        # release — the Unload node rides this seat.
        release_warmed_handles()

    # ── Voice Studio: export / preview / load_qvoice ────────────────────
    #
    # These methods implement the .qvoice file format end-to-end:
    #
    #   export_voice()   — generate a new voice artifact
    #     lite   : VoiceDesign → create_voice_clone_prompt → write_lite
    #     wdelta : VoiceDesign → patch CV talker with Base weights → write_wdelta
    #
    #   preview_voice()  — load a .qvoice and synthesize with it
    #     lite   : load Base, build VoiceClonePromptItem from stored items
    #     wdelta : load CV, apply patched talker_state, generate with
    #              embedding + instruct (the "combo pipeline")
    #
    #   load_qvoice()    — read a .qvoice file and return (kind, payload_dict)
    #
    # The engine bypasses ManagedModel for export (it needs to swap
    # variants multiple times) but the active self.model is whatever
    # was last loaded — so preview reuses the current session when the
    # variant matches, and only reloads when it has to.

    def export_voice(
        self,
        name: str,
        instruct: str,
        *,
        ref_audio: str = "",
        ref_text: str = "",
        speaker: str = "",
        style_instruction: str = "",
        sample_text: str = "",
        wdelta: bool = False,
        voices_dir: str = "",
        language: str = "auto",
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        seed: int = 0,
    ) -> tuple[str, list[float], int]:
        """Generate a .qvoice file; return (absolute path, sample PCM, sr).

        The PCM is the exact design sample that seeded the .qvoice —
        the caller surfaces it as the preview audio instead of stub
        silence (commission 032's live finding).

        Flow (lite):
          1. Load VoiceDesign variant.
          2. generate_voice_design(instruct, text) → ref WAV bytes.
          3. Load Base variant.
          4. create_voice_clone_prompt(ref_audio=wav_bytes) → items.
          5. write_lite(items, instruct, sample_text) → /<voices_dir>/<name>.qvoice.

        Flow (wdelta): as lite, plus:
          6. Load CustomVoice variant.
          7. Patch talker.text_proj + talker.token_embd with Base's weights.
          8. write_wdelta(talker_state_dict) → /<voices_dir>/<name>.qvoice.

        ``name`` is auto-numbered on collision (foo → foo_2 → foo_3).
        """
        from .. import qvoice as _qv

        # VENDOR RE-CURE: the bank rides the wired path or the
        # ``qvoice`` folder seat (never a literal, never env).
        voices_dir = voices_dir or _qv.voices_home()
        os.makedirs(voices_dir, exist_ok=True)
        unique_name = _qv.next_available_name(name, voices_dir)
        out_path = os.path.join(voices_dir, unique_name + ".qvoice")

        synth_text = sample_text or (
            "Hello, I am a friendly voice assistant. "
            "How can I help you today?"
        )
        ref_path = (ref_audio or "").strip()
        transcript = (ref_text or "").strip()
        speaker_name = (speaker or "").strip()
        if speaker_name and ref_path:
            raise RuntimeError(
                "export_voice: state ONE voice source — speaker (preset "
                "save) or reference_audio (clone save), not both")
        if speaker_name and instruct.strip():
            raise RuntimeError(
                "export_voice: state ONE voice source — speaker (preset "
                "save) or instruct (design save), not both")
        if ref_path and instruct.strip():
            raise RuntimeError(
                "export_voice: state ONE voice source — reference_audio "
                "(clone save) or instruct (design save), not both")
        if ref_path and wdelta:
            raise RuntimeError(
                "export_voice: a reference save rides lite (wdelta patches "
                "the DESIGNED talker — a cloned voice has no design)")
        if speaker_name and wdelta:
            raise RuntimeError(
                "export_voice: a preset save rides lite (wdelta patches "
                "the DESIGNED talker — a preset speaker has no design)")
        if not ref_path and not instruct.strip() and not speaker_name:
            raise RuntimeError(
                "export_voice: nothing to save — the save renders with "
                "the press's own params and persists THAT voice (state a "
                "preset speaker, a reference clip, or a design word — "
                "the run path's own one-voice law)"
            )
        style_word = (style_instruction or "").strip()
        if style_word and not speaker_name:
            logger.warning(
                "export_voice: style_instruction conditions only the "
                "CustomVoice render — this save's lane has no speaker "
                "to deliver it; persisting the voice without the style"
            )

        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, seed=seed,
        )

        # THE CLONE-SAVE ARM (the qvoice boolean law, operator
        # 2026-10-22: "save the embeddings to media" from ANY lane):
        # the operator's reference IS the voice — the Base variant
        # extracts the clone prompt directly (x-vector-only when no
        # transcript rides, the mode _voice_clone cures); no
        # VoiceDesign fire. The save preview speaks synth_text in the
        # cloned voice — the sample that seeded the .qvoice, the
        # export arm's own receipt law.
        if ref_path:
            self._ensure_variant("base")
            items = self._qwen.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=transcript or None,
                x_vector_only_mode=not transcript,
            )
            if not items:
                raise RuntimeError(
                    "export_voice: create_voice_clone_prompt returned no "
                    f"items from {ref_path!r}")
            audios, sr = self.model.generate_voice_clone(
                text=synth_text,
                language=language,
                voice_clone_prompt=[items[0]],
                **gen_kwargs,
            )
            sample_pcm, sample_sr = self._to_pcm(audios, sr)
            payload_size = _qv.write_lite(
                out_path,
                name=unique_name,
                instruct="",
                sample_text=synth_text,
                items=items,
                export_variant=_qv.size_tag(self._model_dir or ""),
            )
            logger.info(
                "qvoice: wrote lite %s from reference %s (%d bytes payload)",
                out_path, ref_path, payload_size,
            )
            return out_path, sample_pcm, sample_sr

        # THE PRESET-SAVE ARM (the qvoice boolean law, third lane —
        # operator 2026-10-22: "save the embeddings to media" from
        # ANY lane): the preset speaker IS the voice — CustomVoice
        # renders it speaking synth_text, that render is the
        # reference (its transcript is known verbatim — full ICL,
        # never x-vector-only), Base extracts the clone prompt, lite
        # persists. The preview speaks in the preset's voice.
        if speaker_name:
            self._ensure_variant("customvoice")
            # Style conditions the RENDER (the CV lane's instruct slot —
            # _custom_voice passes instruct through exactly as stated);
            # the extraction takes the voice from what style delivered.
            audios, sr = self.model.generate_custom_voice(
                text=synth_text, speaker=speaker_name,
                language=language, instruct=style_word or None,
                **gen_kwargs,
            )
            sample_pcm, sample_sr = self._to_pcm(audios, sr)
            ref_wav_path = self._write_wav(sample_pcm, sample_sr)
            try:
                self._ensure_variant("base")
                items = self._qwen.create_voice_clone_prompt(
                    ref_audio=ref_wav_path,
                    ref_text=synth_text,
                    x_vector_only_mode=False,  # transcript known verbatim
                )
                if not items:
                    raise RuntimeError(
                        "create_voice_clone_prompt returned no items "
                        f"from the preset {speaker_name!r} render")
                payload_size = _qv.write_lite(
                    out_path,
                    name=unique_name,
                    instruct="",
                    sample_text=synth_text,
                    items=items,
                    export_variant=_qv.size_tag(self._model_dir or ""),
                )
                logger.info(
                    "qvoice: wrote lite %s from preset speaker %s "
                    "(%d bytes payload)",
                    out_path, speaker_name, payload_size,
                )
                return out_path, sample_pcm, sample_sr
            finally:
                if ref_wav_path and os.path.exists(ref_wav_path):
                    os.unlink(ref_wav_path)

        # Step 1: VoiceDesign → ref WAV.
        self._ensure_variant("voicedesign")
        ref_wav_path, sample_pcm, sample_sr = self._render_voicedesign_to_wav(
            instruct=instruct, text=synth_text, language=language,
            gen_kwargs=gen_kwargs,
        )

        try:
            # Step 2: Base variant → create_voice_clone_prompt → items.
            self._ensure_variant("base")
            # ICL mode (x_vector_only_mode=False) requires the reference
            # transcript — and we KNOW it verbatim: the ref WAV was just
            # rendered by VoiceDesign speaking synth_text.
            items = self._qwen.create_voice_clone_prompt(
                ref_audio=ref_wav_path,
                ref_text=synth_text,
                x_vector_only_mode=False,  # we want full ICL items (ref_code + emb)
            )
            if not items:
                raise RuntimeError(
                    "create_voice_clone_prompt returned no items"
                )

            if not wdelta:
                payload_size = _qv.write_lite(
                    out_path,
                    name=unique_name,
                    instruct=instruct,
                    sample_text=synth_text,
                    items=items,
                    export_variant=_qv.size_tag(self._model_dir or ""),
                )
                logger.info(
                    "qvoice: wrote lite %s (%d bytes payload)",
                    out_path, payload_size,
                )
                return out_path, sample_pcm, sample_sr

            # Step 3 (wdelta only): patch CV talker with Base weights.
            base_text_proj, base_token_embd = self._extract_base_patch_tensors()
            # Free Base before loading CV — VRAM headroom.
            self._drop_model()
            self._ensure_variant("customvoice")
            talker_state = self._patch_cv_with_base(
                base_text_proj=base_text_proj,
                base_token_embd=base_token_embd,
            )

            payload_size = _qv.write_wdelta(
                out_path,
                name=unique_name,
                instruct=instruct,
                talker_state=talker_state,
                text_proj_state=base_text_proj,
                token_embd_state=base_token_embd,
                source_ref_audio=ref_wav_path,
                sample_text=synth_text,
            )
            logger.info(
                "qvoice: wrote wdelta %s (%d bytes payload)",
                out_path, payload_size,
            )
            return out_path, sample_pcm, sample_sr
        finally:
            # Clean up the temp ref WAV; it's only needed during export.
            try:
                os.unlink(ref_wav_path)
            except OSError:
                pass

    def preview_voice(
        self,
        qvoice_path: str,
        text: str,
        *,
        instruct: str = "",
        language: str = "auto",
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        voice_strength: float = 1.0,
        speed: float = 1.0,
        emotion: str = "",
        seed: int = 0,
    ) -> tuple[list[float], int]:
        """Synthesize text with a .qvoice file. Returns (pcm, sample_rate).

        Dispatches on the qvoice kind:
          lite   → load Base, build VoiceClonePromptItem from stored items,
                   call generate_voice_clone. The stored ref_code +
                   ref_spk_embedding drive ICL cloning.
          wdelta → load CV, apply patched talker_state, re-extract embedding
                   from the stored source_ref_audio (or use stored), then
                   call generate_voice_clone with embedding + instruct.
                   This is the "combo pipeline" — speaker identity AND
                   style instruction simultaneously.
        """
        from .. import qvoice as _qv

        if not os.path.isfile(qvoice_path):
            raise RuntimeError(f"preview_voice: no such file: {qvoice_path}")

        kind = _qv.detect_kind(qvoice_path)
        if kind is None:
            raise RuntimeError(
                f"preview_voice: not a .qvoice file: {qvoice_path}"
            )

        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, seed=seed,
        )
        lang = self._coerce_language(language)
        # Voice strength scales the embedding's deviation from the mean —
        # values <1.0 pull toward neutral, >1.0 exaggerate identity. We
        # apply it at the tensor level before building the prompt.
        strength = float(max(0.0, min(2.0, voice_strength)))

        if kind == "lite":
            return self._preview_lite(
                qvoice_path, text, instruct, lang, gen_kwargs, strength,
            )
        return self._preview_wdelta(
            qvoice_path, text, instruct, lang, gen_kwargs, strength,
            emotion, speed,
        )

    def load_qvoice(self, path: str) -> dict[str, Any]:
        """Read a .qvoice file and return its payload as a dict.

        Adds ``kind`` to the payload so callers don't need to call
        detect_kind separately. The payload schema is documented in
        qvoice.py (lite vs wdelta).
        """
        from .. import qvoice as _qv

        kind = _qv.detect_kind(path)
        if kind is None:
            raise RuntimeError(f"load_qvoice: not a .qvoice file: {path}")
        payload = _qv.read(path)
        payload["kind"] = kind
        return payload

    # ── Voice Studio internals ──────────────────────────────────────────

    def _bank_neutral_embedding(self, emb: Any, qvoice_path: str) -> Any:
        """The voices-bank mean speaker embedding — the 'neutral' anchor.

        The bank is the rendered artifact's own directory; every readable
        lite .qvoice sibling donates its items' ref_spk_embedding at THIS
        voice's dimension (a foreign checkpoint's artifact, a wdelta, or
        an unreadable file is not a bank member — skipped, never fatal).
        x-vector mean normalization: the average voice is the neutral the
        docs promise ("below 1 pulls toward neutral").

        Cached per (bank dir, dim) until the bank's newest mtime moves, so
        a long-lived ComfyUI process does not re-read the bank per render.

        :param emb: this voice's speaker embedding (the dim probe).
        :param qvoice_path: the rendered artifact (its dir IS the bank).
        :returns: the mean embedding tensor, or None when no same-dim
            sibling embedding is readable (the caller rides identity).
        """
        import torch
        from .. import qvoice as _qv

        try:
            dim = int(tuple(emb.shape)[-1]) if isinstance(emb, torch.Tensor) else len(emb)
        except Exception:
            return None
        bank = os.path.dirname(qvoice_path) or "."
        try:
            entries = sorted(os.listdir(bank))
        except OSError:
            return None
        newest = 0.0
        for name in entries:
            if name.endswith(".qvoice"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(bank, name)))
                except OSError:
                    pass
        key = (bank, dim)
        cached = _BANK_NEUTRAL_CACHE.get(key)
        if cached is not None and cached[0] == newest:
            return cached[1]
        acc = None
        count = 0
        for name in entries:
            if not name.endswith(".qvoice"):
                continue
            try:
                payload = _qv.read_lite(os.path.join(bank, name))
            except Exception:
                continue
            for item in (payload.get("items") or []):
                vec = (item or {}).get("ref_spk_embedding")
                if vec is None:
                    continue
                if not isinstance(vec, torch.Tensor):
                    try:
                        vec = torch.tensor(vec, dtype=torch.float32)
                    except Exception:
                        continue
                if vec.numel() != dim:
                    continue
                vec = vec.reshape(-1).to(torch.float32)
                acc = vec.clone() if acc is None else acc + vec
                count += 1
        mean = None if acc is None or count == 0 else (acc / count).to(torch.float32)
        _BANK_NEUTRAL_CACHE[key] = (newest, mean)
        return mean

    def _steer_identity_strength(self, emb: Any, strength: float,
                                 qvoice_path: str) -> Any:
        """voice_strength as an IDENTITY SCALE — the F51 cure (2026-11-01).

        The old ``emb * strength`` rescaled the raw embedding, and the
        codec-conditioned clone pipeline answered a 0.5x / 1.4x embedding
        with a wholesale new performance (locked-seed corr 0.10 — the
        audit's F51 re-roll). The PCA steering law (AudiocoreTTS
        ``voice_pca_strengths``: ``emb + direction * strength``) is the
        mechanism instead: steer along the voice's own identity axis —

            steered = neutral + (emb - neutral) * strength

        1.0 → exactly the saved embedding (zero delta: the voice LOCK is
        bit-identical, the short-circuit below makes that true by
        construction), < 1 pulls toward the bank's neutral voice, > 1
        exaggerates — the identity scale the card documents.

        THE RENDER TRUTH (the live receipt, 2026-11-01): the speaker
        embedding sits in the talker's codec prefill (qwen_tts
        core/models/modeling_qwen3_tts.py generate loop), so ANY
        conditioning move re-samples the code stream — the performance
        re-rolls at every strength ≠ 1, at every sampling temperature,
        at every steering magnitude (measured: decoded-PCM corr ≈ 0 for
        0.9-vs-1.1, for 0.5-vs-1.4, at temperature 0.9 AND 0.05; the
        same voice re-rendered at a different seed also corr ≈ 0 — raw
        PCM correlation is a bit-identity detector here, not a
        similarity metric; the seed-locked same-graph control is
        bit-identical). The steering law therefore buys a MEANINGFUL
        timbre axis — the voice actually moves toward neutral /
        exaggerates — not a locked performance; that is the honest
        semantic the docs state.

        No bank mean (empty/foreign-dim bank) degrades to identity with a
        loud warning: strength rides as 1.0, never a silent re-roll.

        :param emb: the item's speaker embedding.
        :param strength: the card's voice_strength (already clamped 0..2).
        :param qvoice_path: the artifact (its dir is the neutral's bank).
        :returns: the steered embedding (the input object at 1.0 / degrade).
        """
        import torch
        if strength == 1.0:
            return emb
        if isinstance(emb, torch.Tensor):
            emb = emb.to(torch.float32)
        neutral = self._bank_neutral_embedding(emb, qvoice_path)
        if neutral is None:
            logger.warning(
                "preview_voice: no bank mean for %s (empty or foreign-dim "
                "voices bank) — voice_strength %.2f rides as 1.0 (identity), "
                "never a re-roll.",
                qvoice_path, strength,
            )
            return emb
        return neutral + (emb - neutral) * strength

    def _preview_lite(
        self, qvoice_path: str, text: str, instruct: str,
        lang: str, gen_kwargs: dict[str, Any], strength: float,
    ) -> tuple[list[float], int]:
        from .. import qvoice as _qv
        from qwen_tts import VoiceClonePromptItem
        import torch

        payload = _qv.read_lite(qvoice_path)

        # ARTIFACT-DRIVEN VARIANT (the 1024/2048 catch, 2026-09-22):
        # a lite .qvoice is size-locked to its export checkpoint —
        # the stored export_variant names it; legacy payloads fall
        # back to the embedding's own length (1024 → 0.6B, 2048 →
        # 1.7B). NEVER the size-less "base" (the repo-order default
        # would load 1.7B against a 1024-dim artifact and die
        # mid-graph).
        hint = _qv.preview_size_hint(payload)
        if hint is None:
            raise RuntimeError(
                "preview_voice: lite qvoice carries no size "
                "provenance (no export_variant, unreadable embedding "
                "length) — re-export the voice."
            )
        self._ensure_variant(hint)
        items_raw = payload.get("items") or []
        if not items_raw:
            raise RuntimeError(
                "preview_voice: lite qvoice has no items — re-export."
            )

        # Reconstruct VoiceClonePromptItem list. Apply voice_strength as
        # an identity scale (the F51 cure): steer each embedding along
        # its deviation from the bank's neutral voice — never the raw
        # `emb * strength` rescale that re-rolled the whole performance.
        items: list[Any] = []
        for raw in items_raw:
            emb = raw.get("ref_spk_embedding")
            if emb is not None and strength != 1.0:
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                emb = self._steer_identity_strength(emb, strength, qvoice_path)
            items.append(VoiceClonePromptItem(
                ref_code=raw.get("ref_code"),
                ref_spk_embedding=emb,
                x_vector_only_mode=bool(raw.get("x_vector_only_mode", False)),
                icl_mode=bool(raw.get("icl_mode", False)),
                ref_text=raw.get("ref_text"),
            ))

        # Lite voices support instruct via the Base variant's instruct input.
        gen = dict(gen_kwargs)
        if instruct:
            gen["instruct"] = instruct
        audios, sr = self.model.generate_voice_clone(
            text=text, language=lang, voice_clone_prompt=items, **gen,
        )
        return self._to_pcm(audios, sr)

    def _preview_wdelta(
        self, qvoice_path: str, text: str, instruct: str,
        lang: str, gen_kwargs: dict[str, Any], strength: float,
        emotion: str, speed: float,
    ) -> tuple[list[float], int]:
        from .. import qvoice as _qv

        # WDELTA needs the CV variant with the patched talker applied.
        self._ensure_variant("customvoice")
        payload = _qv.read_wdelta(qvoice_path)

        talker_state = payload.get("talker_state") or {}
        if talker_state:
            applied = self._apply_talker_state(talker_state)
            if not applied:
                # NEVER SILENT (2026-09-22): an unapplied talker patch
                # means the rendered voice is NOT the designed one —
                # the artifact is from another model size. Warn-and-
                # continue rendered a wrong voice as if it were right.
                raise RuntimeError(
                    "preview_voice: wdelta qvoice's talker_state did not "
                    "match any module parameters — the artifact is from "
                    "a different model size than the loaded "
                    "CustomVoice. Re-export the voice on this "
                    "checkpoint's size."
                )

        # Re-extract the embedding from the stored source_ref_audio so
        # voice_strength has a meaningful base. The wdelta path's whole
        # point is "speaker embedding + instruct simultaneously".
        ref_audio = payload.get("source_ref_audio") or ""
        items: list[Any] = []
        # ICL mode needs the reference transcript; write_wdelta stored the
        # sample text the source ref audio actually speaks. Absent → None →
        # the upstream loud ValueError (unchanged behavior, never silent).
        icl_ref_text = str(payload.get("sample_text") or "")
        if ref_audio and os.path.isfile(ref_audio):
            items = self._qwen.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=icl_ref_text or None,
                x_vector_only_mode=False,
            )
        elif payload.get("text_proj_state"):
            # No ref audio available — fall back to a degenerate embedding
            # (zeros). The patched talker will still produce the designed
            # voice identity because the text_proj + token_embd carry the
            # VoiceDesign signature.
            logger.warning(
                "preview_voice: wdelta qvoice has no source_ref_audio; "
                "using zero embedding (voice identity comes from the "
                "patched talker only)."
            )

        gen = dict(gen_kwargs)
        if instruct:
            gen["instruct"] = instruct
        if emotion:
            # Faster's generate_* take no emotion kwarg — drop it LOUD
            # rather than TypeErroring the whole preview.
            logger.warning(
                "preview_voice: wdelta qvoice requests emotion=%r, which "
                "the faster backend does not condition on — rendering "
                "without it (instruct + patched talker still apply).",
                emotion,
            )

        if items:
            # Apply voice_strength as an identity scale (the F51 cure):
            # the same bank-neutral steering the lite path rides — never
            # the raw rescale that re-rolled the performance.
            if strength != 1.0:
                import torch
                for it in items:
                    emb = getattr(it, "ref_spk_embedding", None)
                    if isinstance(emb, torch.Tensor):
                        it.ref_spk_embedding = self._steer_identity_strength(
                            emb, strength, qvoice_path)
            audios, sr = self.model.generate_voice_clone(
                text=text, language=lang,
                voice_clone_prompt=items, **gen,
            )
        else:
            # No ref audio — use custom_voice with the patched talker.
            # instruct still applies; emotion was already dropped loud
            # above (no backend param carries it). speaker="vivian" is
            # STATED here (the combo path owns its choice), never a default.
            audios, sr = self.model.generate_custom_voice(
                text=text, speaker="vivian", language=lang, **gen,
            )

        # Speed is applied as a post-processing step (neither backend
        # exposes it in generate()). We do a no-op pass for speed == 1.0.
        pcm, sr_out = self._to_pcm(audios, sr)
        if abs(speed - 1.0) > 0.001:
            pcm = self._resample_linear(pcm, speed)
        return pcm, sr_out

    def _render_voicedesign_to_wav(
        self, *, instruct: str, text: str, language: str,
        gen_kwargs: dict[str, Any],
    ) -> tuple[str, list[float], int]:
        """Run VoiceDesign generation, write the output to a temp WAV.

        Returns ``(temp_path, pcm, sample_rate)`` — the PCM rides
        along so the export caller can surface the REAL design
        sample as the run's preview audio (the preview flac was
        1-sample silence, 0.000042s, while the engine had just
        rendered the full sample — the live-lane finding of
        commission 032). The caller owns unlinking the temp path.
        """
        audios, sr = self.model.generate_voice_design(
            text=text, language=language, instruct=instruct, **gen_kwargs,
        )
        pcm, sr = self._to_pcm(audios, sr)
        return self._write_wav(pcm, sr), pcm, sr

    @staticmethod
    def _write_wav(pcm: list[float], sr: int) -> str:
        """Write PCM float32 samples to a 16-bit WAV in /tmp. Returns path."""
        import wave
        import struct
        fd, path = tempfile.mkstemp(prefix="audiocore_voicedesign_",
                                    suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                # Clamp + convert to int16.
                import numpy as np
                arr = np.clip(np.asarray(pcm, dtype=np.float32),
                              -1.0, 1.0)
                i16 = (arr * 32767.0).astype("<i2").tobytes()
                # wave.Wave_write API needs a wave writer, not the raw
                # BufferedWriter os.fdopen returned (AttributeError:
                # '_io.BufferedWriter' object has no attribute
                # 'setnchannels' — broke every VoiceDesign render and
                # therefore every .qvoice export, 2026-08-25).
                with wave.open(f, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sr)
                    wav.writeframes(i16)
        except Exception:
            os.unlink(path)
            raise
        return path

    @staticmethod
    def _resample_linear(pcm: list[float], speed: float) -> list[float]:
        """Time-stretch PCM by `speed` via simple linear resampling.

        speed > 1.0 → faster (shorter). speed < 1.0 → slower (longer).
        This is a no-frills resampler — quality is acceptable for preview.
        The final render uses no native rate parameter (neither backend
        exposes one).
        """
        if not pcm or abs(speed - 1.0) < 0.001:
            return pcm
        n_out = max(1, int(len(pcm) / speed))
        idx = [int(i * speed) for i in range(n_out)]
        return [pcm[min(i, len(pcm) - 1)] for i in idx]

    def _ensure_variant(self, variant: str) -> None:
        """Adopt the warmed handle for `variant`. Bypasses ManagedModel.

        The handle comes from the process-wide cache (loaded + warmed
        once at first use, reused by every later run) — never a fresh
        load per call. Uses the same _resolve_model_dir helper so HF
        cache lookup, flat-dir detection, and snapshot_download
        fallback all work.
        """
        if self.model is not None and self._variant == variant:
            return
        # Release this engine's claim before adopting the cached
        # handle. The cache keeps the weights resident for the next
        # run — _drop_model only drops OUR reference, never the
        # cached copy.
        if self.model is not None:
            self._drop_model()
        # Use a synthetic path: _resolve_model_dir keys off the variant hint.
        # An empty path forces the HF repo lookup path.
        hf_dir = self._resolve_model_dir("", variant_hint=variant)
        if hf_dir is None:
            raise RuntimeError(
                f"qwen3_tts: variant {variant!r} is not available locally. "
                f"Download the matching Qwen3-TTS HF dir "
                f"({self._hf_repo_for_variant(variant)}) and place it under "
                f"the `qwen3_tts` model folder declared in "
                f"extra_model_paths.yaml "
                f"(image-gen/comfyui/audio/qwen3-tts/hf/)."
            )
        self.model = _load_warmed_handle(hf_dir, variant)
        self._model_dir = hf_dir
        self._variant = variant
        logger.info("qwen3_tts: variant=%s ready (device=%s)",
                    variant, self.model.device)

    def _drop_model(self) -> None:
        """Release the current model's VRAM without touching _variant."""
        if self.model is None:
            return
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        try:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _detect_variant_from_dir(hf_dir: str) -> str:
        """Infer the variant from the HF directory name suffix."""
        d = os.path.basename(hf_dir.rstrip("/")).lower()
        if "voicedesign" in d:
            return "voicedesign"
        if "base" in d:
            return "base"
        return "customvoice"

    @staticmethod
    def _hf_repo_for_variant(variant: str) -> str:
        return {
            "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }.get(variant, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

    def _extract_base_patch_tensors(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pull text_proj + token_embd state_dict slices from the loaded Base.

        These are the tensors that differ between Base (which accepts a
        continuous speaker embedding) and CustomVoice (which takes a
        discrete speaker id). Applying them to a CV talker turns it into
        a hybrid that accepts BOTH inputs.
        """
        if self.model is None or self._variant != "base":
            raise RuntimeError(
                "_extract_base_patch_tensors: Base variant must be loaded"
            )
        import torch
        talker = self._get_talker()
        if talker is None:
            raise RuntimeError(
                "could not locate talker submodule on the loaded model"
            )
        sd = talker.state_dict()
        text_proj = {
            k: v.detach().cpu().clone()
            for k, v in sd.items()
            if "text_proj" in k or "lm_head" in k
        }
        token_embd = {
            k: v.detach().cpu().clone()
            for k, v in sd.items()
            if "token_embd" in k or "embed_tokens" in k
        }
        if not text_proj or not token_embd:
            raise RuntimeError(
                f"could not find text_proj/token_embd tensors in Base talker "
                f"(keys seen: {list(sd.keys())[:10]}...)"
            )
        return text_proj, token_embd

    def _patch_cv_with_base(
        self, *, base_text_proj: dict[str, Any],
        base_token_embd: dict[str, Any],
    ) -> dict[str, Any]:
        """Patch the loaded CV talker with Base's text_proj + token_embd.

        Returns the full patched talker state_dict (CPU tensors) so it
        can be written to disk via qvoice.write_wdelta.
        """
        if self.model is None or self._variant != "customvoice":
            raise RuntimeError(
                "_patch_cv_with_base: CustomVoice variant must be loaded"
            )
        talker = self._get_talker()
        if talker is None:
            raise RuntimeError("could not locate talker submodule")
        sd = talker.state_dict()
        # Apply only matching keys — shape mismatch raises a RuntimeError
        # that the caller can surface as a model-size mismatch.
        patched = 0
        for k, v in {**base_text_proj, **base_token_embd}.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v.to(sd[k].device, dtype=sd[k].dtype)
                patched += 1
        if patched == 0:
            raise RuntimeError(
                "no Base tensors matched the CV talker — model sizes differ. "
                "Ensure both variants are the same B-parameter size (e.g. "
                "both 0.6B or both 1.7B)."
            )
        logger.info(
            "qvoice: patched %d tensors in CV talker with Base weights",
            patched,
        )
        # Move to CPU for portable serialization.
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def _apply_talker_state(self, talker_state: dict[str, Any]) -> bool:
        """Load a patched talker state_dict into the live model.

        Returns True if at least one tensor was applied. Tensors that
        don't match a module key or shape are skipped (with a debug log)
        so preview doesn't crash on minor architecture drift.
        """
        talker = self._get_talker()
        if talker is None:
            return False
        import torch
        current = talker.state_dict()
        applied = 0
        to_load: dict[str, torch.Tensor] = {}
        for k, v in talker_state.items():
            if k not in current:
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if v.shape != current[k].shape:
                logger.debug(
                    "qvoice: skip %s (shape %s != %s)",
                    k, tuple(v.shape), tuple(current[k].shape),
                )
                continue
            to_load[k] = v.to(current[k].device, dtype=current[k].dtype)
            applied += 1
        if applied == 0:
            return False
        # Use load_state_dict with strict=False so missing/extra keys are
        # tolerated — we already filtered to matching keys.
        talker.load_state_dict(to_load, strict=False)
        logger.info("qvoice: applied %d talker tensors from .qvoice", applied)
        return True

    def _get_talker(self) -> Any:
        """Locate the talker sub-module on the wrapped qwen_tts model.

        FasterQwen3TTS.model IS a qwen_tts.Qwen3TTSModel, which wraps
        Qwen3TTSForConditionalGeneration exposing ``.talker`` (the LM)
        and ``.codec`` (the audio decoder). Different package versions
        name these differently, so we probe common paths.
        """
        m = self._qwen
        # Direct attribute (newer qwen_tts).
        for attr in ("talker", "lm_model", "model"):
            sub = getattr(m, attr, None)
            if sub is not None and hasattr(sub, "state_dict"):
                return sub
        # Some versions wrap under .qwen3_tts.
        inner = getattr(m, "qwen3_tts", None)
        if inner is not None:
            for attr in ("talker", "lm_model", "model"):
                sub = getattr(inner, attr, None)
                if sub is not None and hasattr(sub, "state_dict"):
                    return sub
        return None

    @staticmethod
    def _clamp_max_new_tokens(raw: Any) -> int:
        """Clamp the decode length to the static-cache budget.

        Ported from ComfyUI-Qwen3-TTS-Fast's ``_clamp_tokens``: the
        captured graphs hold ``max_seq_len`` positions, so a request
        beyond ``max_seq_len - 256`` cannot decode — clamp LOUD (log
        the clamp) instead of decoding into unheld cache.
        """
        try:
            want = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            want = _MAX_NEW_TOKENS
        if want > _MAX_NEW_TOKENS:
            logger.warning(
                "qwen3_tts: max_new_tokens %d exceeds the static-cache "
                "budget; clamping to %d.", want, _MAX_NEW_TOKENS)
            return _MAX_NEW_TOKENS
        return max(1, want)

    @staticmethod
    def _build_gen_kwargs(
        *, temperature: float, top_p: float, top_k: int,
        repetition_penalty: float, seed: int,
        max_new_tokens: int = _MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Build the gen kwargs forwarded to Faster's generate_*.

        seed conditions the torch RNG only and is NEVER forwarded —
        Faster's generate_* take no seed kwarg.
        """
        out: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": Qwen3TtsEngine._clamp_max_new_tokens(
                max_new_tokens),
        }
        if int(seed) > 0:
            import torch
            torch.manual_seed(int(seed))
        return out

    # Alias so ManagedModel lifecycle (which calls .close() on the C++
    # session) works uniformly on either kind of _session.
    def close(self) -> None:
        self.unload()

    # ── dispatch helpers ────────────────────────────────────────────────

    def _custom_voice(self, text: str, speaker: str, language: str,
                      instruct: str, gen_kwargs: dict[str, Any]
                      ) -> tuple[list[float], int]:
        # Unstated speaker is a LOUD refuse — never a silent preset.
        # (The wdelta combo path below passes speaker="vivian"
        # explicitly; that call site owns its stated choice.)
        if not speaker:
            raise RuntimeError(
                "qwen3_tts: plain TTS needs a stated speaker preset "
                "(e.g. speaker='ryan') or a .qvoice — refusing to "
                "render with a silent default voice."
            )
        # The 0.6B CustomVoice model ignores instruct (upstream's own
        # 0b6 guard is dead code — len-4 size never in len-3 "0b6" —
        # so we pass instruct through exactly as stated).
        audios, sr = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct or None,
            **gen_kwargs,
        )
        return self._to_pcm(audios, sr)

    def _voice_clone(self, text: str, language: str,
                     ref_audio: str, ref_text: str,
                     kwargs: dict[str, Any], gen_kwargs: dict[str, Any]
                     ) -> tuple[list[float], int]:
        # Prefer a pre-computed speaker_embedding (from AudiocoreVoiceEmbedding).
        emb = kwargs.get("speaker_embedding")
        if emb is not None:
            # Build a VoiceClonePromptItem from the embedding so we skip
            # re-extracting from a reference WAV.
            try:
                prompt_item = self._build_prompt_from_embedding(emb)
                # generate_voice_clone takes a LIST of VoiceClonePromptItem.
                audios, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=[prompt_item],
                    **gen_kwargs,
                )
                return self._to_pcm(audios, sr)
            except Exception as e:
                logger.warning(
                    "qwen3_tts: embedding-based clone failed (%s); "
                    "falling back to ref_audio path", e)

        if not ref_audio:
            raise RuntimeError(
                "qwen3_tts voice_clone requires either speaker_embedding "
                "(from AudiocoreVoiceEmbedding) or reference_audio path."
            )
        # No transcript stated → x-vector-only clone (the embedding
        # lane), never Faster's ICL default: qwen_tts raises
        # "ref_text is required when x_vector_only_mode=False" for a
        # transcript-less reference (the MP3 burn's second act). The
        # same mode compute_embedding rides; ref_audio loads in
        # Python (MP3 included).
        if not (ref_text or "").strip():
            prompt_items = self._qwen.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=None,
                x_vector_only_mode=True,
            )
            if not prompt_items:
                raise RuntimeError(
                    "qwen3_tts: voice clone produced no prompt item from "
                    f"{ref_audio!r}")
            audios, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=[prompt_items[0]],
                **gen_kwargs,
            )
            return self._to_pcm(audios, sr)
        audios, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text or None,
            **gen_kwargs,
        )
        return self._to_pcm(audios, sr)

    def _voice_design(self, text: str, language: str, instruct: str,
                      gen_kwargs: dict[str, Any]
                      ) -> tuple[list[float], int]:
        audios, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **gen_kwargs,
        )
        return self._to_pcm(audios, sr)

    def _build_prompt_from_embedding(self, emb: Any) -> Any:
        """Wrap a pre-computed speaker embedding into a VoiceClonePromptItem."""
        from qwen_tts import VoiceClonePromptItem
        import torch
        import numpy as np
        # Unwrap the AUDIOCORE_EMBEDDING envelope from the ComfyUI node.
        if isinstance(emb, dict) and "vector" in emb:
            emb = emb["vector"]
        if isinstance(emb, list):
            emb = torch.tensor(emb, dtype=torch.float32)
        elif isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        elif isinstance(emb, torch.Tensor):
            emb = emb.float()
        else:
            raise RuntimeError(
                f"qwen3_tts: unsupported speaker_embedding type "
                f"{type(emb).__name__}"
            )
        # VoiceClonePromptItem fields (per qwen_tts source):
        #   ref_code, ref_spk_embedding, x_vector_only_mode, icl_mode, ref_text
        # x_vector_only_mode + no ref_code → use the embedding directly.
        return VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=emb,
            x_vector_only_mode=True,
            icl_mode=False,
            ref_text=None,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_pcm(audios: list, sr: int) -> tuple[list[float], int]:
        """Convert generate_*'s List[np.ndarray] → (list[float], int)."""
        import numpy as np
        if not audios:
            return [], int(sr)
        a = audios[0]
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        return a.tolist(), int(sr)

    @staticmethod
    def _coerce_seed(seed: Any) -> int:
        """audiocore convention: seed=0 → randomize."""
        try:
            s = int(seed)
        except (TypeError, ValueError):
            s = 0
        if s <= 0:
            import random
            return random.randint(1, 2_147_483_647)
        return s

    @staticmethod
    def _coerce_language(lang: Any) -> str:
        """Map short codes → full names expected by the backend.

        Faster forwards language to qwen, which accepts: auto, chinese,
        english, french, german, italian, japanese, korean, portuguese,
        russian, spanish. Map common short forms; pass through unknown
        values (the backend will validate).
        """
        if not lang:
            return "auto"
        s = str(lang).strip().lower()
        return {
            "en": "english", "zh": "chinese", "cn": "chinese",
            "fr": "french", "de": "german", "it": "italian",
            "ja": "japanese", "jp": "japanese",
            "ko": "korean", "kr": "korean",
            "pt": "portuguese", "br": "portuguese",
            "ru": "russian", "es": "spanish",
        }.get(s, s)

    @staticmethod
    def _resolve_model_dir(path: str, variant_hint: str | None = None) -> str | None:
        """Resolve `path` to an HF source dir containing config.json.

        Variant detection (in priority order):
          1. `variant_hint` arg ('base' / 'customvoice' / 'voicedesign')
             — set from the `variant` key in extras JSON.
          2. Path suffix: '*-base', '*-voicedesign', '*-customvoice'.

        The backend gates features per variant (same matrix under
        Faster — it forwards to qwen):
          - customvoice  → generate_custom_voice (default TTS)
          - base         → generate_voice_clone + create_voice_clone_prompt
          - voicedesign  → generate_voice_design

        Handles four inputs:
          1. Absolute HF dir → return as-is if it has config.json.
          2. Absolute GGUF dir (qwen3-tts) → look for HF source nearby.
          3. Relative name "qwen3-tts" / "qwen3_tts" → search candidates.
          4. Relative HF dir name → resolve via the declared model folders
             (extra_model_paths.yaml — the `audiocpp` / `qwen3_tts` folders).
        """
        roots = [
            # VENDOR RE-CURE: folder_paths categories only, never env.
            resolve_model_folder("audiocpp"),
            resolve_model_folder("qwen3_tts"),
        ]

        # Determine which variant to load.
        want_variant = "customvoice"  # default
        want_size = None  # None = 1.7B first (the quality preference)
        if variant_hint:
            hint = variant_hint.lower()
            if "base" in hint or "clone" in hint:
                want_variant = "base"
            elif "design" in hint:
                want_variant = "voicedesign"
            elif "custom" in hint or "voice" in hint:
                want_variant = "customvoice"
            # SIZE PROVENANCE (2026-09-22, the 1024/2048 catch): a
            # .qvoice artifact is size-locked to the checkpoint that
            # exported it (0.6B speaker encoder emits 1024-dim
            # embeddings, 1.7B emits 2048 — the talker cats them and
            # a mismatch dies mid-graph). Hints may carry the size
            # ("base-0.6b"/"base-1.7b"): when stated it ORDERS the
            # repo list; when absent the 1.7B leads (quality).
            if "0.6" in hint:
                want_size = "0.6B"
            elif "1.7" in hint:
                want_size = "1.7B"
        else:
            # Fall back to path-suffix detection.
            path_lc = path.lower().rstrip("/")
            tail = os.path.basename(path_lc).replace("-", "_").replace(".", "_")
            if "base" in tail or "voiceclone" in tail or "voice_clone" in tail:
                want_variant = "base"
            elif "design" in tail:
                want_variant = "voicedesign"

        # Per-variant HF repo IDs (for snapshot_download lookup).
        # LARGER FIRST (operator catch f, 2026-09-22): the resolver
        # walks this list and takes the first variant present
        # locally — 0.6B-first silently condemned the clone/reuse
        # lanes to the small model's weaker fidelity (worse persona
        # lock) even where the 1.7B dirs exist. 1.7B leads; 0.6B is
        # the fallback when the larger dir was never placed. (The
        # dirs live under extra_model_paths' qwen3_tts/hf — they are
        # placed out-of-band, not provisioned file-by-file.)
        variant_hf_repos = {
            "base": [
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            ],
            "customvoice": [
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            ],
            "voicedesign": [
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            ],
        }[want_variant]
        # NO SIZE FALLBACK: an artifact locked to one size CANNOT
        # render on the other (the talker cat would die mid-graph) —
        # an empty list makes the resolver return None and the
        # loader's loud "download the matching dir" error names it.
        if want_size:
            variant_hf_repos = [
                r for r in variant_hf_repos if want_size in r
            ]

        candidates: list[str] = []
        if os.path.isabs(path):
            candidates.append(path)
        else:
            for root in roots:
                candidates.append(os.path.join(root, path))

        # Only honor absolute/relative candidates that match the requested
        # variant (so a GGUF dir at /mnt/data/models/audio/qwen3-tts doesn't
        # shadow the Base variant when variant_hint='base').
        want_suffix = {
            "base": "-base",
            "voicedesign": "-voicedesign",
            "customvoice": "-customvoice",
        }[want_variant]
        if variant_hint:
            filtered = [c for c in candidates if want_suffix in c.lower()]
            for c in filtered:
                if os.path.isfile(os.path.join(c, "config.json")):
                    return c
        else:
            for c in candidates:
                if os.path.isfile(os.path.join(c, "config.json")):
                    return c

        # Variant-specific HF cache lookup. The repo short-name (after the
        # last "/") appears in two layouts under <hf_root>:
        #   1. Flat dir (manual download):
        #        <hf_root>/Qwen3-TTS-12Hz-0.6B-CustomVoice/config.json
        #   2. HF cache layout (snapshot_download):
        #        <hf_root>/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/<hash>/config.json
        # REPO ORDER DOMINATES LAYOUT (2026-09-22, the measured
        # catch): the original two-pass shape (all flat dirs first,
        # then all snapshots) let a SMALLER flat dir beat a LARGER
        # snapshot — the 1.7B-Base landed via snapshot_download and
        # the resolver still loaded the 0.6B-Base flat dir. Each repo
        # now checks flat THEN snapshot before yielding to the next.
        hf_root = roots[1]
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            snapshot_download = None
        for repo_id in variant_hf_repos:
            short = repo_id.split("/")[-1]
            flat = os.path.join(hf_root, short)
            if os.path.isfile(os.path.join(flat, "config.json")):
                return flat
            if snapshot_download is None:
                continue
            try:
                resolved = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=hf_root,
                    local_files_only=True,  # don't hit network at inference time
                )
                if resolved and os.path.isfile(os.path.join(resolved, "config.json")):
                    return resolved
            except Exception:
                continue
        return None
