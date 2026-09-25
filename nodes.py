"""ComfyUI node classes for the audiocpp-fork audio engine.

Exposes the full inference surface via NATIVE in-process loading
(libaudiocore_native.so loaded by ctypes — no HTTP, no subprocess):
  - LoadAudiocoreModel  — load a family (moss_tts_nano / moss_tts_local / ace_step / qwen3_tts / moss_sfx_v2)
  - AudiocoreTTS        — full TTS (voice clone, design, multilingual) AND SFX (moss_sfx_v2)
  - AudiocoreMusic      — text-to-music (ACE-Step)
  - AudiocoreVoiceEmbedding — speaker embedding extraction
  - UnloadAudiocoreModel — release VRAM
  - AudiocoreFamilyInfo — list registered families
  - AudiocoreVoiceStudio — voice artifact authoring (uses qwen-tts Python directly)
  - MeliteAudioSlice — the audio-bed cut node (sample-grain slicing +
    silence-pad; the h3-timeline convenience layer's engine half)
- MeliteFilmAudioMix — the valve's second arm (2026-09-24): the film's
    post-generation audio mix (window files + positioned clips + duck
    spans, torch only — lives in film_mix.py, registered here)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import numpy as np
import torch

from .core import ManagedModel, _AUDIOCPP_MODELS_DIR
from .film_mix import MeliteFilmAudioMix
from .qvoice_copy import MeliteQVoiceCopy
from .loudness import MeliteLoudness

logger = logging.getLogger("audiocore-nodes")


# ── ComfyUI progress plumbing ────────────────────────────────────────────────
#
# The native engine_runtime (loaded via ctypes) installs a progress callback
# on the session (IVoiceTaskSession::set_progress_callback) which the C++ model
# code fires at natural milestones: per diffusion step for moss_sfx_v2 (step
# 0/25, step 10/25, …), per pipeline phase for ACE-Step, per text chunk for
# TTS. run_tts_streaming / run_music_streaming forward those REAL callbacks
# verbatim. NO FALLBACK: if no real progress arrives, nothing is emitted
# (2026-08-10 exterminate-the-fallbacks policy — the old time-budget tick
# fabricated totals from wall-clock ceilings ("step 3/900") and emitted
# monotonic float elapsed as the value). ComfyUI's own execution machinery
# stamps running → finished, so a silent node still shows its lifecycle —
# just no fake numbers.
#
# The mechanism is ComfyUI v0.30's ProgressRegistry (comfy_execution/progress).
# update_progress() notifies WebUIProgressHandler, which emits a progress_state
# frame carrying the REGISTRY's prompt_id — so we never have to plumb prompt_id
# ourselves. The sync node FUNCTION runs in the execution task (no
# run_in_executor: execution.py:296 runs f(**inputs) inline under
# CurrentNodeContext), so get_executing_context() on this thread returns our
# node_id. _emit_progress is always called from THIS thread; the worker thread
# only runs the native inference call.


def _emit_progress(value: float, max_value: float) -> None:
    """Best-effort progress_state update via ComfyUI's ProgressRegistry.

    No-op outside a prompt execution (tests, standalone) and never raises —
    telemetry must not break the node. Routes through the registry rather than
    a bare send_sync so prompt_id is filled by the registry itself.
    """
    try:
        from comfy_execution.progress import get_progress_state
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        if ctx is None or ctx.node_id is None:
            return
        registry = get_progress_state()
        if registry is None:
            return
        # Clamp value to max: the engine's load ratio (bytes loaded / total)
        # can overshoot 1.0 by epsilon (GGUF counting) and finish_progress
        # stamps value = max verbatim — a bar must never exceed 100%.
        value = min(float(value), float(max_value))
        registry.update_progress(ctx.node_id, value, float(max_value))
    except Exception:
        pass


def _run_with_progress(fn, *, interval: float = 1.0):
    """Run ``fn(report)`` in a worker thread while forwarding progress from
    the node's execution thread.

    ``report(step, total)`` is handed to ``fn`` so the streaming audiocpp
    calls can forward REAL per-chunk / per-phase progress straight off the
    server's SSE feed. The main thread emits the latest real value each
    ``interval``.

    NO FALLBACK (2026-08-10, design rationale: exterminate all fallbacks):
    if no real progress arrives, nothing is emitted — a silent node stays
    silent. The previous time-budget tick lied: it reported the wall-clock
    ceiling (900 s) as the TOTAL (so the UI showed "step 3/900" — the
    "900 steps" bug) and elapsed monotonic time as the VALUE (the
    ``1.0000807540200185`` float that crashed the client's int progress
    model). ComfyUI's own execution machinery already stamps
    running → finished (execution.py start_progress/finish_progress), so
    the node's lifecycle is never invisible — only fake numbers are gone.
    Native note (2026-08-11): the progress callback fires from the C++
    inference thread inside libaudiocore_native.so — the engine reports
    real per-step / per-phase milestones (e.g. moss_sfx_v2 emits
    step/total at each diffusion step, visible in the logs).

    Returns fn()'s result, or re-raises its exception here.
    """
    latest: dict = {"step": None, "total": None}

    def report(step, total) -> None:
        latest["step"], latest["total"] = step, total

    result_box: dict = {}
    error_box: list = []
    last_sent: tuple | None = None

    def _worker() -> None:
        try:
            result_box["value"] = fn(report)
        except BaseException as exc:  # re-raised on the calling thread
            error_box.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        thread.join(timeout=interval)
        step, total = latest["step"], latest["total"]
        # Dedupe: the final report lands in the join that returns at thread
        # death — without last_sent, the last value would broadcast twice.
        if step is not None and (step, total) != last_sent:
            _emit_progress(step, total)
            last_sent = (step, total)
        if not thread.is_alive():
            break
    if error_box:
        raise error_box[0]
    return result_box["value"]


def _load_progress_sender():
    try:
        from comfy_execution.utils import get_executing_context
        from server import PromptServer

        ctx = get_executing_context()
        server = PromptServer.instance
        client_id = getattr(server, "client_id", None)
        if ctx is None or ctx.node_id is None or not client_id:
            return None
        node_id = ctx.node_id
        return lambda message: server.send_progress_text(message, node_id, client_id)
    except Exception:
        return None


def _run_with_load_progress(fn):
    sender = _load_progress_sender()

    def report(message):
        if sender is None or not isinstance(message, str) or message == "":
            return
        try:
            sender(message)
        except Exception:
            pass

    result_box: dict = {}
    error_box: list = []

    def _worker() -> None:
        try:
            result_box["value"] = fn(report)
        except BaseException as exc:
            error_box.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box[0]
    return result_box["value"]


FAMILY_NAMES = {
    "moss_tts_nano": "MOSS-TTS Nano (8B, Unigram tokenizer)",
    "moss_tts_local": "MOSS-TTS Local (8B, BPE tokenizer)",
    "qwen3_tts": "Qwen3-TTS (1.7B)",
    "ace_step": "ACE-Step (music)",
    "moss_sfx_v2": "MOSS-SFX v2 (sound effects)",
}

_DEFAULT_MODEL_DIR = {
    "moss_tts_nano": "moss-tts",
    "moss_tts_local": "moss-tts",
    "qwen3_tts": "qwen3-tts",
    "ace_step": "acestep-cpp-converted",
    # Torch checkpoint (model_index.json) — GGUF dirs are retired for this
    # family (2026-08-11: the pure-torch pipeline replaces the C++ engine).
    "moss_sfx_v2": "MOSS-SoundEffect-v2.0-src",
}


_WEIGHT_FILE_NAMES = ("model.safetensors", "model_index.json")
_WEIGHT_FILE_SUFFIXES = (".gguf",)
# Walk bound: the provisioning tree nests at most a few levels (the
# HF cache layout is <name>/snapshots/<hash>/); unbounded recursion
# over a shared models mount is a boot-time hazard.
_MAX_WALK_DEPTH = 5


def _dir_holds_weights(abs_dir: str) -> bool:
    try:
        names = os.listdir(abs_dir)
    except OSError:
        return False
    for n in names:
        if n in _WEIGHT_FILE_NAMES:
            return True
        if n.endswith(_WEIGHT_FILE_SUFFIXES):
            return True
    return False


# Support artifacts, never lane addresses: the speech tokenizer rides
# INSIDE its model dir (its own weight files make it a false choice),
# and blobs/ are the HF cache's raw internals.
_SKIP_DIR_NAMES = {"speech_tokenizer", "blobs", "refs", "__pycache__"}


def _list_audiocore_models() -> list[str]:
    """The combo enum serves the REAL provisioning tree (2026-09-24,
    transcript-015): top-level dirs stay choices (legacy behavior —
    container dirs like qwen3-tts/ ride even without direct weights),
    and every nested dir that DIRECTLY holds weights is a choice too —
    the HF cache layout (<root>/<name>/snapshots/<hash>/model.safetensors)
    is how the qwen3-tts lanes ship. Before this, ComfyUI's combo
    validation refused every nested lane address at /prompt (400)
    even though _resolve_model_path would have joined it fine."""
    found: list[str] = []
    root = _AUDIOCPP_MODELS_DIR

    def walk(rel: str, depth: int) -> None:
        abs_dir = os.path.join(root, rel) if rel else root
        try:
            entries = sorted(os.listdir(abs_dir))
        except OSError:
            return
        for e in entries:
            if e in _SKIP_DIR_NAMES:
                continue
            rel_child = f"{rel}/{e}" if rel else e
            abs_child = os.path.join(root, rel_child)
            if not os.path.isdir(abs_child):
                continue
            # Top-level dirs are choices regardless (kept: the legacy
            # enum + container dirs); a nested dir is a choice only
            # when it DIRECTLY holds weights.
            if not rel or _dir_holds_weights(abs_child):
                found.append(rel_child)
            # Nested dirs recurse whether or not they hold weights
            # directly (their children may — the snapshot layout).
            if depth < _MAX_WALK_DEPTH:
                walk(rel_child, depth + 1)

    try:
        walk("", 0)
    except OSError:
        return []
    return sorted(set(found))


def _resolve_model_path(model_path: str) -> str:
    if os.path.isabs(model_path):
        return model_path
    try:
        import folder_paths
        resolved = folder_paths.get_full_path("audiocore", model_path)
        if resolved and os.path.exists(resolved):
            return resolved
    except ImportError:
        pass
    return os.path.join(_AUDIOCPP_MODELS_DIR, model_path)


def _resolve_input_path(value: str) -> str:
    """Resolve a file input to a container-absolute path.

    Reference audio / .qvoice files arrive as bare ComfyUI input-dir
    filenames (uploaded via /upload/image — the same convention LoadImage
    uses) OR as absolute host paths (drag-and-drop from the assets
    sidebar, e.g. /mnt/data/models/audio/voices/Cherry.qvoice). The
    engine's os.path.isfile() checks run against the CONTAINER filesystem,
    so bare names must be joined with ComfyUI's input directory here —
    the one boundary every entry path (/v1/run, pipeline families, raw
    API) crosses. Unresolvable values pass through unchanged; the engine
    raises its own clear error.
    """
    if not value or os.path.isfile(value):
        return value
    try:
        import folder_paths
        candidate = os.path.join(folder_paths.get_input_directory(), value)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    return value


# ── Node: Load Audiocore Model ───────────────────────────────────────────────

class LoadAudiocoreModel:
    """Load an audiocore model.

    moss_sfx_v2 → pure-torch diffusion pipeline (from_pretrained).
    Other families → native in-process C++ session (libaudiocore_native.so).
    """

    TITLE = "Load Audiocore Model"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIOCORE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        models = _list_audiocore_models()
        default_family = "moss_tts_nano"
        default_model = _DEFAULT_MODEL_DIR.get(default_family, "")
        if default_model not in models and models:
            default_model = models[0]
        return {
            "required": {
                "family": (list(FAMILY_NAMES.keys()),
                           {"default": default_family}),
                "model_path": (models,
                               {"default": default_model}),
            },
            "optional": {
                # JSON object from the builder, e.g.
                # {"variant": "CustomVoice"}. Consumed by ManagedModel to
                # disambiguate which GGUF to load when the family directory
                # holds multiple variants:
                #   qwen3_tts: Base / CustomVoice / VoiceDesign
                #   ace_step:  turbo / sft   (substring-matched on the
                #              filename, so {"variant":"sft"} resolves to
                #              ace-step-1.5-sft-q8_0.gguf)
                # Empty string = use the model_specs default (largest GGUF
                # in the dir via sidecar exclusion — currently the turbo pkg).
                "extras": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": 'JSON e.g. {"variant":"sft"} for ACE-Step SFT, {"variant":"turbo"} for turbo. Empty = default.',
                }),
            },
        }

    def load(self, family: str, model_path: str, extras: str = ""):
        resolved_path = _resolve_model_path(model_path)
        extras_dict: dict = {}
        if extras:
            try:
                parsed = json.loads(extras)
                if isinstance(parsed, dict):
                    extras_dict = parsed
            except ValueError:
                logger.warning("LoadAudiocoreModel: ignoring bad extras JSON: %s", extras)
        m = ManagedModel(family, resolved_path, extras=extras_dict)
        if not _run_with_load_progress(lambda report: m.load(on_progress=report)):
            raise RuntimeError(f"Failed to load {family} from {resolved_path}")
        return (m,)


# ── Node: Audiocore TTS ──────────────────────────────────────────────────────

class AudiocoreTTS:
    """Text-to-speech AND sound-effect generation via the native engine_runtime.

    Modes (family-dependent):
      tts    — plain text-to-speech
      clone  — zero-shot voice cloning (needs reference_audio + reference_text)
      design — instruction-following voice design (needs instruct)

    For moss_sfx_v2 (sound effects, task="gen"), the diffusion params
    guidance_scale / num_inference_steps / duration_seconds drive the
    diffusion loop — they reach the engine's options map via
    _build_speech_request (no silent drops — every param forwards).

    Voice files (.voice) — pre-computed speaker embeddings loaded and PCA-steered
    in the node, then passed to the engine as speaker_embedding.
    """

    TITLE = "Audiocore TTS"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                # NOTE: the input is named ``prompt`` — the pipeline's
                # universal field name (catalog SCHEMA → family signature
                # → builder → node, ONE vocabulary). The vendored audio-core
                # fork called it ``text``; we renamed OUR fork so no
                # translation layer exists (2026-08-06, Phase D1).
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello world.",
                }),
            },
            "optional": {
                "mode": (["tts", "clone", "design"],
                         {"default": "tts"}),
                "voice": ("STRING", {"default": ""}),
                "language": ("STRING", {
                    "default": "",
                    "placeholder": "en, zh, auto...",
                }),
                "temperature": ("FLOAT",
                                {"default": 0.8, "min": 0.0, "max": 2.0,
                                 "step": 0.05}),
                "top_p": ("FLOAT",
                          {"default": 0.9, "min": 0.0, "max": 1.0,
                           "step": 0.01}),
                "top_k": ("INT",
                          {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "speed": ("FLOAT",
                          {"default": 1.0, "min": 0.5, "max": 2.0,
                           "step": 0.1}),
                "repetition_penalty": ("FLOAT",
                                       {"default": 1.05, "min": 0.8,
                                        "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 2147483647}),
                # ── SFX diffusion params (moss_sfx_v2, task="gen") ──
                # These reach the engine's diffusion loop via
                # _build_speech_request → options map. Ignored by TTS
                # families (the engine reads only what it needs).
                "guidance_scale": ("FLOAT",
                                   {"default": 5.0, "min": 0.0, "max": 20.0,
                                    "step": 0.1,
                                    "tooltip": "Diffusion classifier-free guidance (SFX only)"}),
                "num_inference_steps": ("INT",
                                        {"default": 50, "min": 1, "max": 500,
                                         "step": 1,
                                         "tooltip": "Diffusion steps (SFX only — more = higher quality, slower)"}),
                "duration_seconds": ("FLOAT",
                                     {"default": 10.0, "min": 0.5, "max": 300.0,
                                      "step": 0.5,
                                      "tooltip": "Output length in seconds (SFX only)"}),
                "reference_audio": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/clone_ref.wav (mode=clone)",
                }),
                "reference_text": ("STRING", {
                    "default": "",
                    "placeholder": "transcript of reference_audio",
                }),
                "speaker_name": ("STRING", {"default": ""}),
                "instruct": ("STRING", {
                    "default": "",
                    "placeholder": "emotion/style direction (works with ANY mode)",
                    "multiline": True,
                }),
                "speaker_embedding": ("AUDIOCORE_EMBEDDING",),
                "voice_file": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/voice.voice (pre-computed speaker embedding)",
                }),
                "voice_pca_strengths": ("STRING", {
                    "default": "",
                    "placeholder": '{"pca_pc1.dir": 0.5, "pca_pc2.dir": -0.3}',
                    "multiline": True,
                }),
            },
        }

    def synthesize(self, model: ManagedModel, prompt: str, **kwargs):
        if not hasattr(model, "run_tts"):
            raise RuntimeError("invalid model reference")

        call_kwargs = {
            k: v for k, v in kwargs.items()
            if v is not None and v != ""
        }

        # Resolve file inputs against ComfyUI's input dir — bare uploaded
        # filenames become container-absolute paths the engine can stat.
        if call_kwargs.get("reference_audio"):
            call_kwargs["reference_audio"] = _resolve_input_path(
                call_kwargs["reference_audio"])
        if call_kwargs.get("voice_file"):
            call_kwargs["voice_file"] = _resolve_input_path(
                call_kwargs["voice_file"])

        # "none" = NO preset speaker (operator 2026-09-24): the card offers
        # it so a CustomVoice run doesn't force a default timbre (Ryan/
        # Vivian/…). Normalized to EMPTY here — the engine boundary — so
        # the identity comes from whatever actually rides (a .qvoice via
        # VoiceStudio's torch path), or the engine refuses loud when
        # nothing does. Never a silent fallback to a preset.
        if call_kwargs.get("voice") == "none":
            call_kwargs["voice"] = ""

        # ── Voice file loading + PCA steering ──
        voice_file = call_kwargs.pop("voice_file", "")
        pca_json = call_kwargs.pop("voice_pca_strengths", "")

        if voice_file:
            import json as _json
            import struct as _struct
            import numpy as _np

            with open(voice_file, "rb") as f:
                data = f.read()
            MAGIC = b"QWEN3VOICE"
            if len(data) >= 36 and data[:len(MAGIC)] == MAGIC:
                dim = _struct.unpack_from("<I", data, 20)[0]
                emb = _np.frombuffer(data, dtype=_np.float32,
                                     count=dim, offset=32)
            else:
                emb = _np.frombuffer(data, dtype=_np.float32)
            emb = _np.array(emb, dtype=_np.float32)

            if pca_json:
                import os.path as _osp
                voices_dir = _osp.dirname(voice_file)
                strengths = _json.loads(pca_json)
                for dir_name, strength in strengths.items():
                    dir_path = _osp.join(voices_dir, dir_name)
                    if not _osp.exists(dir_path):
                        continue
                    with open(dir_path, "rb") as f:
                        ddata = f.read()
                    if len(ddata) >= 36 and ddata[:len(MAGIC)] == MAGIC:
                        ddim = _struct.unpack_from("<I", ddata, 20)[0]
                        direction = _np.frombuffer(ddata, dtype=_np.float32,
                                                   count=ddim, offset=32)
                    else:
                        direction = _np.frombuffer(ddata, dtype=_np.float32)
                    direction = _np.array(direction, dtype=_np.float32)
                    if len(direction) == len(emb):
                        emb = emb + direction * float(strength)

            call_kwargs.pop("speaker_embedding", None)
            call_kwargs["speaker_embedding"] = emb.tolist()

        # ── Mode alias mapping ──
        mode = call_kwargs.get("mode", "tts")
        has_voice = bool(call_kwargs.get("reference_audio")
                         or call_kwargs.get("speaker_embedding")
                         or call_kwargs.get("voice_path"))
        if mode == "clone" or (has_voice and mode in ("tts", "design", "")):
            call_kwargs["mode"] = "voice_clone"

        # Native inference: the first request after load absorbs CUDA graph
        # warmup (~20 s for moss_sfx_v2). Subsequent requests are compute-only
        # and ~2.5× faster (measured 8.6 s vs 21.6 s — the model persists
        # in GPU memory between generations). The native progress callback
        # fires real per-step / per-phase milestones; no time-budget
        # fallback (2026-08-10 exterminate-the-fallbacks policy).
        pcm, sr = _run_with_progress(
            lambda report: model.run_tts_streaming(
                prompt, on_progress=report, **call_kwargs,
            ),
        )
        audio_np = np.clip(np.array(pcm, dtype=np.float32), -1.0, 1.0)
        waveform = torch.from_numpy(audio_np).reshape(1, 1, -1)
        return ({"waveform": waveform, "sample_rate": sr},)


# ── Node: Audiocore Voice Embedding ──────────────────────────────────────────

class AudiocoreVoiceEmbedding:
    """Compute a speaker embedding from a WAV file (voice caching)."""

    TITLE = "Audiocore Voice Embedding"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIOCORE_EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "compute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                "wav_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/voice.wav",
                }),
            },
        }

    def compute(self, model: ManagedModel, wav_path: str):
        if not wav_path:
            raise RuntimeError("wav_path is required")
        emb = _run_with_progress(
            lambda report: model.compute_embedding(wav_path),
                    )
        if not emb:
            raise RuntimeError(
                "compute_embedding returned empty — "
                "only qwen3_tts with a loaded speaker_encoder GGUF "
                "supports this call"
            )
        return ({"vector": torch.tensor(emb, dtype=torch.float32)},)


# ── Node: Audiocore Voice Studio ──────────────────────────────────────────────

class AudiocoreVoiceStudio:
    """Voice Studio — create and preview .qvoice voice artifacts.

    Uses the qwen-tts Python package directly (not the C++ server) because
    voice export/preview involves loading multiple model variants and
    extracting/patching tensors — operations the HTTP API doesn't support.
    """

    TITLE = "Voice Studio"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                "mode": (
                    ["export_lite", "export_wdelta", "preview", "generate"],
                    {"default": "preview"},
                ),
            },
            "optional": {
                "name": ("STRING", {
                    "default": "",
                    "placeholder": "voice name (auto-numbered on collision)",
                }),
                "instruct": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "A warm female voice with a slight British accent.",
                }),
                "sample_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Defaults to a friendly greeting.",
                }),
                "voices_dir": ("STRING", {
                    "default": "",
                    "placeholder": "/mnt/data/models/audio/voices (default)",
                }),
                "qvoice_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/voice.qvoice OR bare name (Cherry)",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is a voice preview.",
                }),
                "language": ("STRING", {
                    "default": "",
                    "placeholder": "en, zh, auto (default)",
                }),
                "emotion": ("STRING", {
                    "default": "",
                    "placeholder": "happy, sad, angry, neutral (wdelta only)",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 0, "max": 200,
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 0.8, "max": 2.0, "step": 0.01,
                }),
                "voice_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
        }

    def run(self, model: ManagedModel, mode: str, **kwargs):
        # Import the Python engine directly for Voice Studio operations.
        try:
            from .engines.qwen3_tts import Qwen3TtsEngine as _Qwen3TtsEngine
        except ImportError as e:
            raise RuntimeError(
                "Voice Studio requires the qwen-tts Python package: "
                f"{e}"
            ) from e

        engine = _Qwen3TtsEngine()
        # Resolve voice_dir from the model path
        model_dir = os.path.dirname(model.path) if os.path.isfile(
            os.path.join(model.path, "config.json")
        ) else model.path
        kw = {k: v for k, v in kwargs.items() if v not in (None, "")}

        if mode in ("export_lite", "export_wdelta"):
            name = kw.get("name") or "voice"
            instruct = kw.get("instruct") or ""
            if not instruct.strip():
                raise RuntimeError(
                    f"{mode}: instruct is required — describe the voice"
                )
            out_path, sample_pcm, sample_sr = _run_with_progress(
                lambda report: engine.export_voice(
                    name=name,
                    instruct=instruct,
                    sample_text=kw.get("sample_text") or "",
                    wdelta=(mode == "export_wdelta"),
                    voices_dir=kw.get("voices_dir") or "",
                    language=kw.get("language") or "auto",
                    temperature=float(kw.get("temperature", 0.9)),
                    top_p=float(kw.get("top_p", 1.0)),
                    top_k=int(kw.get("top_k", 50)),
                    repetition_penalty=float(kw.get("repetition_penalty", 1.05)),
                    seed=int(kw.get("seed", 0)),
                ),
                            )
            # THE REAL DESIGN SAMPLE (commission 032's live finding,
            # fixed 2026-09-24): the export just rendered the sample
            # that seeded the .qvoice — return IT as the preview
            # waveform (the old 1-sample silence stub reported
            # 0.000042s while real audio existed). Same clamp/reshape
            # law as the preview arm; empty PCM (a defensive engine
            # arm) alone falls back to the stub.
            if sample_pcm:
                audio_np = np.clip(
                    np.array(sample_pcm, dtype=np.float32), -1.0, 1.0,
                )
                waveform = torch.from_numpy(audio_np).reshape(1, 1, -1)
            else:
                waveform = torch.zeros(1, 1, 1, dtype=torch.float32)
                sample_sr = 24000
            return {
                "ui": {"qvoice_path": [out_path], "mode": [mode]},
                "result": (
                    {"waveform": waveform, "sample_rate": sample_sr},
                ),
            }

        qvoice_path = _resolve_input_path(kw.get("qvoice_path") or "")
        if not qvoice_path:
            raise RuntimeError(f"{mode}: qvoice_path is required")
        text = kw.get("text") or ""
        if not text:
            raise RuntimeError(f"{mode}: text is required")

        pcm, sr = _run_with_progress(
            lambda report: engine.preview_voice(
                qvoice_path,
                text,
                instruct=kw.get("instruct") or "",
                language=kw.get("language") or "auto",
                temperature=float(kw.get("temperature", 0.9)),
                top_p=float(kw.get("top_p", 1.0)),
                top_k=int(kw.get("top_k", 50)),
                repetition_penalty=float(kw.get("repetition_penalty", 1.05)),
                voice_strength=float(kw.get("voice_strength", 1.0)),
                speed=float(kw.get("speed", 1.0)),
                emotion=kw.get("emotion") or "",
                seed=int(kw.get("seed", 0)),
            ),
                    )
        audio_np = np.clip(np.array(pcm, dtype=np.float32), -1.0, 1.0)
        waveform = torch.from_numpy(audio_np).reshape(1, 1, -1)
        return ({"waveform": waveform, "sample_rate": sr},)


# ── Node: Audiocore Music ────────────────────────────────────────────────────

class AudiocoreMusic:
    """Text-to-music generation via ACE-Step (native in-process engine_runtime)."""

    TITLE = "Audiocore Music"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                # ``prompt`` — the pipeline's universal field name (same
                # rename as AudiocoreTTS; the audio-core fork called this
                # ``caption``. ONE vocabulary end to end, 2026-08-06).
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "lo-fi ambient piano",
                }),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("FLOAT",
                             {"default": 30.0, "min": 1.0, "max": 300.0,
                              "step": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 2147483647}),
                # Diffusion classifier-free guidance. The C++ engine defaults
                # to 7.0 (request_parser.cpp:161). turbo IGNORES this field
                # entirely — diffusion.cpp:1479 gates the CFG path on
                # ``!config.is_turbo`` — so 7.0 is harmless for turbo and
                # correct for base/SFT. The old node default (1.0) silently
                # disabled CFG for base/SFT (1.0 > 1.0 is false →
                # use_diffusion_cfg = false → no guidance → degraded output).
                # Per the ACE-Step 1.5 Musician's Guide: SFT default is 7.0.
                "guidance_scale": ("FLOAT",
                                   {"default": 7.0, "min": 0.1, "max": 10.0,
                                    "step": 0.1,
                                    "tooltip": "Diffusion CFG. turbo ignores this (distilled); base/SFT use it. 7.0 = ACE-Step default. ≤1.0 disables CFG."}),
                "n_diffusion_steps": ("INT",
                                      {"default": 0, "min": 0, "max": 200}),
                "temperature": ("FLOAT",
                                {"default": 0.85, "min": 0.0, "max": 2.0,
                                 "step": 0.05}),
                "top_p": ("FLOAT",
                          {"default": 0.9, "min": 0.0, "max": 1.0,
                           "step": 0.01}),
                "lm_cfg_scale": ("FLOAT",
                                 {"default": 2.0, "min": 0.0, "max": 10.0,
                                  "step": 0.1}),
            },
        }

    def generate(self, model: ManagedModel, prompt: str, **kwargs):
        if not hasattr(model, "run_music"):
            raise RuntimeError("invalid model reference")

        call_kwargs = {
            k: v for k, v in kwargs.items()
            if v is not None and v != ""
        }

        # Music generation runs long (ACE-Step diffusion over minutes). Wrap
        # the blocking native call so the editor sees a live tick for the
        # whole run; the native progress callback fires real per-step /
        # per-phase milestones from the C++ engine.
        pcm, sr, channels = _run_with_progress(
            lambda report: model.run_music_streaming(
                prompt, on_progress=report, **call_kwargs,
            ),
                    )
        audio_t = torch.tensor(pcm, dtype=torch.float32).clamp(-1.0, 1.0)
        waveform = audio_t.reshape(-1, channels).T.unsqueeze(0).contiguous()
        return ({"waveform": waveform, "sample_rate": sr},)


# ── Node: Unload Audiocore Model ─────────────────────────────────────────────

class UnloadAudiocoreModel:
    """Release a model's VRAM. Destroys the native session (libaudiocore_native.so)."""

    TITLE = "Unload Audiocore Model"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ()
    FUNCTION = "unload"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
            },
        }

    def unload(self, model: ManagedModel):
        model.unload()
        return ()


# ── Node: Audiocore Family Info ──────────────────────────────────────────────

class AudiocoreFamilyInfo:
    """List registered families and current session status."""

    TITLE = "Audiocore Family Info"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": ("AUDIOCORE_MODEL",),
            },
        }

    def info(self, model=None):
        families = list(FAMILY_NAMES.keys())
        lines = [f"Registered families: {', '.join(families) or '(none)'}"]
        for f in families:
            display = FAMILY_NAMES.get(f, f)
            lines.append(f"  {f} -> {display}")
        if model is not None and getattr(model, "loaded", False):
            lines.append("")
            lines.append(
                f"Active session: family={model.family} "
                f"path={model.path}"
            )
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}


# ── MeliteAudioSlice (the audio-bed cut node — the h3-timeline
# convenience layer's engine half, 2026-09-20). ONE track spread
# across a film's windows needs per-window CUTS: sample-grain
# slicing (start_seconds at the source's own sample rate, never a
# resample) + silence-pad to the requested duration (H3's AV path
# needs a legal-length waveform even when the bed ends mid-window;
# existing samples stay bit-for-bit). The math adapts
# Songssx/ComfyUI-MiniMaxH3-TimelineDirector's
# _locked_audio_interval (GPL-3.0, credited — the license note
# lives in docs/comfyui/timeline-director-study.md).
try:
    import folder_paths
except ImportError:  # standalone tooling/tests, never inside ComfyUI
    folder_paths = None

try:
    import av
except ImportError:  # standalone tooling/tests never decode files
    av = None


def load_audio_file(path: str):
    """Decode an audio file as (waveform[1, C, N], sample_rate) — the
    AUDIO dict convention, decoded by core LoadAudio's own PyAV law
    (torchaudio.load needs TorchCodec, which this venv does not ship;
    the torchaudio loader died at execution on every bed slice)."""
    if av is None:
        raise RuntimeError("melite-audio-nodes: PyAV (av) is required to decode audio files")
    with av.open(path) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")
        stream = af.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate)
        n_channels = stream.channels
        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)
        if not frames:
            raise ValueError("No audio frames decoded.")
        wav = torch.cat(frames, dim=1)
        if wav.dtype == torch.int16:
            wav = wav.float() / (2 ** 15)
        elif wav.dtype == torch.int32:
            wav = wav.float() / (2 ** 31)
        return wav.unsqueeze(0), sample_rate


class MeliteAudioSlice:
    """Slice a ComfyUI-dir audio file by seconds; pad silence to length."""

    CATEGORY = "audio/melite"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "slice_audio"
    DESCRIPTION = "Cut a bed track at start_seconds for duration_seconds, padding silence when the source runs short."

    @classmethod
    def INPUT_TYPES(cls):
        # THE INPUT-DIR LAW (fixed 2026-09-21): this ComfyUI build has
        # NO 'audio' folder category — folder_paths.get_filename_list
        # ("audio") raises KeyError, which 500'd object_info and killed
        # EVERY graph carrying this node at validation (the audio bed
        # included — dead at submit, silently). Core LoadAudio's own
        # law lists the INPUT DIRECTORY filtered by content type;
        # this node mirrors it exactly (the staged bed/clip filenames
        # live there).
        if folder_paths is None:
            audio_files: list = []
        else:
            input_dir = folder_paths.get_input_directory()
            os.makedirs(input_dir, exist_ok=True)
            audio_files = sorted(
                folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
            )
        return {
            "required": {
                "audio": (audio_files, {"audio upload": "audio"}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "duration_seconds": ("FLOAT", {"default": 10.0, "min": 0.1, "step": 0.01}),
            },
        }

    def slice_audio(self, audio: str, start_seconds: float, duration_seconds: float):
        if folder_paths is None:
            raise RuntimeError("MeliteAudioSlice requires ComfyUI (folder_paths)")
        if duration_seconds <= 0:
            raise ValueError("MeliteAudioSlice: duration_seconds must be positive")
        path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load_audio_file(path)
        # sample-grain slice (never a resample): start lands on the
        # nearest sample at the SOURCE's own rate
        expected = int(round(duration_seconds * sample_rate))
        source_sample = int(round(max(0.0, start_seconds) * sample_rate))
        source_sample = min(max(0, source_sample), int(waveform.shape[-1]))
        cut = waveform[..., source_sample:source_sample + expected]
        if cut.shape[-1] < expected:
            # silence-pad: existing samples stay bit-for-bit, the
            # tail fills to the legal duration
            cut = torch.nn.functional.pad(cut, (0, expected - int(cut.shape[-1])))
        return ({"waveform": cut, "sample_rate": sample_rate},)


# ── MeliteAudioDelay + MeliteAudioMix (the clips-stitch engine half,
# 2026-09-21 — the restore of the h3-timeline Music row's lost
# consumer). Positioned dialogue/music clips must land in the RENDERED
# film, mixed over the final cut at film time, seamless across window
# seams (the old cc_stage_audio_clips/ffmpeg_concat amix, reborn
# in-graph per the engine law — never a harness-side ffmpeg). Delay
# places one clip (head silence at the source's own rate + an exact
# total-duration fit); Mix sums the film track with each placed clip
# (clip resampled to the base rate, shorter side zero-padded, output
# clamped to [-1,1] like every AUDIO producer in this pack).


class MeliteAudioDelay:
    """Place one clip on the film timeline: pad silence at the head,
    fit the total to duration_seconds (trim or silence-pad)."""

    CATEGORY = "audio/melite"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "delay_audio"
    DESCRIPTION = "Delay an AUDIO stream by delay_seconds (head silence, sample-grain at the source rate) and fit the total to duration_seconds exactly (trim tail / pad silence)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "delay_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001}),
                "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001,
                                               "tooltip": "Exact total length (delay + clip). 0 = no fit (delay + the clip's own length)."}),
            },
        }

    def delay_audio(self, audio: dict, delay_seconds: float, duration_seconds: float):
        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        if delay_seconds < 0:
            delay_seconds = 0.0
        pad_samples = int(round(delay_seconds * sr))
        head = torch.zeros(wf.shape[0], wf.shape[1], pad_samples, dtype=wf.dtype)
        out = torch.cat([head, wf], dim=-1)
        if duration_seconds > 0:
            expected = int(round(duration_seconds * sr))
            if out.shape[-1] > expected:
                out = out[..., :expected]
            elif out.shape[-1] < expected:
                out = torch.nn.functional.pad(out, (0, expected - int(out.shape[-1])))
        return ({"waveform": out, "sample_rate": sr},)


class MeliteAudioMix:
    """Sum two AUDIO streams (the film track + one placed clip).

    audio2 (the clip) is resampled to audio1's (the base's) rate when
    they differ; the shorter side is zero-padded to the longer; the
    sum clamps to [-1,1] (the same convention as every AUDIO producer
    in this pack). length_mode 'base' truncates to the BASE length
    (a clip overhanging the film's end never extends the cut);
    'longest' keeps the longer.
    """

    CATEGORY = "audio/melite"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "mix_audio"
    DESCRIPTION = "Mix two AUDIO streams: clip resampled to the base rate, zero-pad to length, sum, clamp. length_mode 'base' never extends the film."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "length_mode": (["base", "longest"], {"default": "base"}),
            },
        }

    def mix_audio(self, audio1: dict, audio2: dict, length_mode: str):
        wf1, sr1 = audio1["waveform"], int(audio1["sample_rate"])
        wf2, sr2 = audio2["waveform"], int(audio2["sample_rate"])
        if sr1 != sr2:
            import torchaudio.functional as AF
            wf2 = AF.resample(wf2, orig_freq=sr2, new_freq=sr1)
        # channel-count match: mono/stereo mismatch broadcasts the mono
        # side by repeating channels (never a silent drop)
        if wf1.shape[1] != wf2.shape[1]:
            if wf1.shape[1] == 1:
                wf1 = wf1.repeat(1, wf2.shape[1], 1)
            elif wf2.shape[1] == 1:
                wf2 = wf2.repeat(1, wf1.shape[1], 1)
            else:
                c = min(wf1.shape[1], wf2.shape[1])
                wf1, wf2 = wf1[:, :c, :], wf2[:, :c, :]
        target = max(int(wf1.shape[-1]), int(wf2.shape[-1]))
        if length_mode == "base":
            target = int(wf1.shape[-1])
        if int(wf1.shape[-1]) < target:
            wf1 = torch.nn.functional.pad(wf1, (0, target - int(wf1.shape[-1])))
        if int(wf2.shape[-1]) < target:
            wf2 = torch.nn.functional.pad(wf2, (0, target - int(wf2.shape[-1])))
        mixed = (wf1[..., :target] + wf2[..., :target]).clamp(-1.0, 1.0)
        return ({"waveform": mixed, "sample_rate": sr1},)


class MeliteAudioDuck:
    """Duck (attenuate) an AUDIO stream during time spans.

    The film's generated voice must yield to a dialogue clip the
    model was conditioned on (else both speak). segments is a
    semicolon list of `start:end` seconds in the AUDIO's own
    timeline; each span drops to gain_db with linear ramps of
    ramp_ms at the edges. Baseline outside spans is unity — the
    film's own sound is untouched where no clip speaks.
    """

    CATEGORY = "audio/melite"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "duck_audio"
    DESCRIPTION = "Attenuate an AUDIO stream during `start:end;start:end` second spans (gain_db, linear ramp_ms edges). The generated track yields to conditioned dialogue clips."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segments": ("STRING", {"default": "", "multiline": False,
                                        "tooltip": "semicolon list of start:end second spans, e.g. '1.0:2.5;4.0:5.0'"}),
                "gain_db": ("FLOAT", {"default": -14.0, "min": -60.0, "max": 0.0, "step": 0.5}),
                "ramp_ms": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 500.0, "step": 1.0}),
                # THE SLOW RELEASE (operator catch c, 2026-09-22: "during
                # dialogue the music gets quiet, then gets loud again") —
                # one 20ms edge on BOTH sides snapped the bed back like a
                # gate. A duck attacks fast, RELEASES slow (the console
                # law): ramp_ms owns the dip's edge, release_ms owns the
                # return (350ms default reads as breathe-in, not pop).
                "release_ms": ("FLOAT", {"default": 350.0, "min": 0.0, "max": 2000.0, "step": 10.0}),
            },
        }

    def duck_audio(self, audio: dict, segments: str, gain_db: float, ramp_ms: float, release_ms: float):
        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        gain = 10.0 ** (gain_db / 20.0)
        n = int(wf.shape[-1])
        env = torch.ones(n, dtype=wf.dtype)
        ramp = max(1, int(round(ramp_ms / 1000.0 * sr)))
        release = max(1, int(round(release_ms / 1000.0 * sr)))
        for seg in (segments or "").split(";"):
            seg = seg.strip()
            if seg == "":
                continue
            parts = seg.split(":")
            if len(parts) != 2:
                raise RuntimeError(f"MeliteAudioDuck: bad segment '{seg}' — want 'start:end' seconds")
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                raise RuntimeError(f"MeliteAudioDuck: bad segment '{seg}' — want numeric seconds")
            if b <= a:
                continue
            ia, ib = max(0, int(round(a * sr))), min(n, int(round(b * sr)))
            if ib <= ia:
                continue
            # THE PER-SPAN LOCAL ENVELOPE (the 2026-09-22 cure): the
            # first release_ms implementation wrote arcs directly
            # into the shared env with minimum() compositing — but
            # minimum() against the span's own gain floor makes the
            # release arc a SILENT NO-OP (minimum keeps the floor;
            # the film measured a 9.8dB pop at the span end because
            # no release ever landed). Correct construction: each
            # span builds its OWN envelope (attack → floor → release
            # arc), and the shared env is the minimum ACROSS spans —
            # a release lifts toward unity exactly where no other
            # span still holds the floor, and overlaps keep the
            # union floor by construction.
            ra = min(ramp, ib - ia)
            rr = min(release, max(0, n - ib) + (ib - ia) // 2)
            local = torch.ones(n, dtype=wf.dtype)
            local[ia:ib] = gain
            if ra > 0:
                local[ia:ia + ra] = torch.linspace(1.0, gain, ra, dtype=wf.dtype)
            if rr > 0:
                right = torch.linspace(gain, 1.0, rr, dtype=wf.dtype)
                # the arc is ONE continuous ramp of length rr that
                # begins tail_in samples before ib and (when the
                # span's tail room was too short) continues past ib
                # — value reaches unity only at the arc's true end,
                # so the inside and spill slices are consecutive
                # windows of `right`, never overlapping ends (the
                # spill slice is right[tail_in:…], continuing from
                # where the inside slice right[:tail_in] stopped)
                tail_in = min(rr, (ib - ia) - ra)
                if tail_in > 0:
                    local[ib - tail_in:ib] = right[:tail_in]
                spill = min(rr - tail_in, n - ib)
                if spill > 0:
                    local[ib:ib + spill] = right[tail_in:tail_in + spill]
            torch.minimum(env, local, out=env)
        env = env.reshape(1, 1, n)
        return ({"waveform": wf * env, "sample_rate": sr},)


# ── Mappings ─────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LoadAudiocoreModel": LoadAudiocoreModel,
    "AudiocoreTTS": AudiocoreTTS,
    "AudiocoreMusic": AudiocoreMusic,
    "AudiocoreVoiceEmbedding": AudiocoreVoiceEmbedding,
    "AudiocoreVoiceStudio": AudiocoreVoiceStudio,
    "UnloadAudiocoreModel": UnloadAudiocoreModel,
    "AudiocoreFamilyInfo": AudiocoreFamilyInfo,
    "MeliteAudioSlice": MeliteAudioSlice,
    "MeliteAudioDelay": MeliteAudioDelay,
    "MeliteAudioMix": MeliteAudioMix,
    "MeliteAudioDuck": MeliteAudioDuck,
    "MeliteFilmAudioMix": MeliteFilmAudioMix,
    "MeliteQVoiceCopy": MeliteQVoiceCopy,
    "MeliteLoudness": MeliteLoudness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudiocoreModel": "Load Audiocore Model",
    "AudiocoreTTS": "Audiocore TTS",
    "AudiocoreMusic": "Audiocore Music",
    "AudiocoreVoiceEmbedding": "Audiocore Voice Embedding",
    "AudiocoreVoiceStudio": "Voice Studio",
    "UnloadAudiocoreModel": "Unload Audiocore Model",
    "AudiocoreFamilyInfo": "Audiocore Family Info",
    "MeliteAudioSlice": "Melite Audio Slice",
    "MeliteAudioDelay": "Melite Audio Delay",
    "MeliteAudioMix": "Melite Audio Mix",
    "MeliteAudioDuck": "Melite Audio Duck",
    "MeliteFilmAudioMix": "Melite Film Audio Mix",
    "MeliteQVoiceCopy": "Melite QVoice Copy",
    "MeliteLoudness": "Melite Loudness (LUFS)",
}
