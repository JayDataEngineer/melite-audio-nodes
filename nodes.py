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


def _list_audiocore_models() -> list[str]:
    try:
        entries = sorted(os.listdir(_AUDIOCPP_MODELS_DIR))
        return [
            e for e in entries
            if os.path.isdir(os.path.join(_AUDIOCPP_MODELS_DIR, e))
        ]
    except OSError:
        return []


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
        # Native load: the C++ engine loads GGUFs + migrates tensors to GPU.
        # No progress fallback (2026-08-10): the load call reports nothing, so
        # nothing is emitted — the node's running→finished lifecycle still shows.
        if not _run_with_progress(lambda report: m.load()):
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
            out_path = _run_with_progress(
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
            silence = torch.zeros(1, 1, 1, dtype=torch.float32)
            return {
                "ui": {"qvoice_path": [out_path], "mode": [mode]},
                "result": (
                    {"waveform": silence, "sample_rate": 24000},
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


# ── Mappings ─────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LoadAudiocoreModel": LoadAudiocoreModel,
    "AudiocoreTTS": AudiocoreTTS,
    "AudiocoreMusic": AudiocoreMusic,
    "AudiocoreVoiceEmbedding": AudiocoreVoiceEmbedding,
    "AudiocoreVoiceStudio": AudiocoreVoiceStudio,
    "UnloadAudiocoreModel": UnloadAudiocoreModel,
    "AudiocoreFamilyInfo": AudiocoreFamilyInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudiocoreModel": "Load Audiocore Model",
    "AudiocoreTTS": "Audiocore TTS",
    "AudiocoreMusic": "Audiocore Music",
    "AudiocoreVoiceEmbedding": "Audiocore Voice Embedding",
    "AudiocoreVoiceStudio": "Voice Studio",
    "UnloadAudiocoreModel": "Unload Audiocore Model",
    "AudiocoreFamilyInfo": "Audiocore Family Info",
}
