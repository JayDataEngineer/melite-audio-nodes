"""audiocore backend for ComfyUI — TORCH-first, native fallback.

Two engine paths behind ONE ManagedModel → node contract:

- **moss_sfx_v2** (sound effects) runs the PURE-TORCH diffusion pipeline
  (TTS-Audio-Suite pattern, 2026-08-11): `MossSoundEffectPipeline.
  from_pretrained()` loads the HF checkpoint directly; ComfyUI manages the
  model lifecycle like any torch model — gc + empty_cache return VRAM. No
  GGUF, no C++, no ctypes, no eviction nodes. See engines/moss_sfx_v2.py.
- **Every other family** (moss_tts, qwen3_tts, ace_step) runs the audiocpp
  C++ engine_runtime in-process (libaudiocore_native.so via ctypes): no
  subprocess, no HTTP. The C++ path persists until torch ports exist.

The C++ path replaced the old HTTP server architecture (2026-08-11): the
moss_sfx_v2 C++ session leaked its 5.4 GB DiT buffer per session destroy
("first gen worked, second crashed") — that family is torch now.
"""
from __future__ import annotations

import glob as _glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .native import NativeSession, NativeError, get_registry
from .schemas import (
    ParamDropError,
    build_music_request as _build_music_request_validated,
    build_speech_request as _build_speech_request_validated,
)

logger = logging.getLogger("audiocore-natives")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_FAMILY_VRAM_ESTIMATE: dict[str, int] = {
    "moss_tts_nano":     9 * 1024 * 1024 * 1024,
    "moss_tts_local":    9 * 1024 * 1024 * 1024,
    "moss_sfx_v2":       4 * 1024 * 1024 * 1024,
    "qwen3_tts":         2 * 1024 * 1024 * 1024,
    "ace_step":          8 * 1024 * 1024 * 1024,
}

# Family → engine task type. "gen" = AudioGeneration (SFX/music), "tts" = TTS.
_FAMILY_TASK: dict[str, str] = {
    "moss_tts_nano":    "tts",
    "moss_tts_local":   "tts",
    "moss_sfx_v2":      "gen",
    "qwen3_tts":        "tts",
    "ace_step":         "gen",
}

# Variant → engine task code (2026-09-20, the 1.7B receipt stroke):
# the C++ gates per CHECKPOINT variant, not per family — a
# VoiceDesign-variant checkpoint refuses task tts LOUD ("Qwen3
# voice design model only supports the VoiceDesign task", task
# code "vdes" per the fork's session.cpp parse). The lane steers
# via extras.variant (the estate's ENGINE_TRUTH key); the family
# map above stays the default.
_VARIANT_TASK: dict[str, str] = {
    "voicedesign":      "vdes",
}

# Families whose engine loader scans the model DIRECTORY for sibling GGUFs.
_DIR_SCAN_FAMILIES = ("moss_sfx_v2",)

# NOTE: _MUSIC_OPTION_MAP, _LANGUAGE_FULL_NAME, and _normalize_language now
# live in schemas.py — they're part of the declarative pydantic contract.
# core.py delegates request building to schemas.build_speech_request /
# schemas.build_music_request (validated, extra='forbid', no silent drops).

# Engine family specs (audiocpp-fork's model_specs/ — a sibling checkout,
# NOT part of this repo). Env override only; empty = specs unavailable and
# every consumer below degrades gracefully (spec={}, model_spec=None).
_AUDIOCPP_MODEL_SPECS = os.environ.get("AUDIOCPP_MODEL_SPECS_DIR", "")

# Spec discovery walk (transcript-016, 2026-09-24): the C++ loader's
# discover_external_model_spec checks <model_path>/model_specs/<family>.json,
# <model_path>/../model_specs/<family>.json, then walks cwd upward — an HF
# cache checkout (models--*/snapshots/<hash>) reaches none of those, so the
# 1.7B snapshot lanes died at load with "install model_specs/qwen3_tts"
# even though the family spec sits three parents up in the provisioning
# tree (…/hf/model_specs/qwen3_tts.json). Mirror the C++ candidate walk
# python-side over the RESOLVED model path's parents and hand the found
# spec file to the session as model_spec_override (the loader accepts a
# file or a directory). Bounded depth; the env override still wins.
_SPEC_WALK_MAX_DEPTH = 6


def _discover_model_spec(family: str, model_path: str) -> Optional[str]:
    if _AUDIOCPP_MODEL_SPECS:
        candidate_dir = Path(_AUDIOCPP_MODEL_SPECS)
        if candidate_dir.is_dir():
            return str(candidate_dir)
        return None
    # Path-style model inputs (a single weights file) anchor at the parent.
    anchor = Path(model_path)
    if anchor.suffix and not anchor.is_dir():
        anchor = anchor.parent
    cursor = anchor
    for _ in range(_SPEC_WALK_MAX_DEPTH):
        candidate = cursor / "model_specs" / f"{family}.json"
        if candidate.is_file():
            return str(candidate)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return None


def resolve_model_folder(folder_key: str, env_name: str) -> str:
    """Resolve one of the pack's model folders DECLARATIVELY via ComfyUI's
    folder_paths, falling back to an env var for standalone/test usage."""
    try:
        import folder_paths
    except ImportError:
        folder_paths = None
    if (
        folder_paths is not None
        and folder_key in folder_paths.folder_names_and_paths
    ):
        paths = folder_paths.get_folder_paths(folder_key)
        if paths:
            return paths[0]
    value = os.environ.get(env_name)
    if value:
        return value
    raise RuntimeError(
        f"model folder {folder_key!r} is not declared: add it to ComfyUI's "
        f"extra_model_paths.yaml or set {env_name} for standalone use"
    )


_AUDIOCPP_MODELS_DIR = resolve_model_folder("audiocpp", "AUDIOCPP_MODELS_DIR")


# ─────────────────────────────────────────────────────────────────────────────
# Model path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model_file(
    family: str, model_path: str, variant: str = "", model_file: str = "",
) -> str:
    """Resolve a model path to the exact path the engine's loader expects.

    Dir-scan families (moss_sfx_v2): the directory IS the path.
    Standalone-GGUF families: resolve to the specific .gguf file.
    Multi-GGUF dirs: pick by variant substring, spec default, or sidecar exclusion.
    ``model_file`` (route-declared basename — builders thread the catalog
    route's required_files entry as extras.model_file) is AUTHORITATIVE
    when given: resolved exactly, with a hard error if missing. Never a
    silent fallback to another weights file — the size/order heuristic
    would load the wrong quant (q8/bf16 lottery) and the mismatch is
    invisible (UI says BF16, engine runs Q8).
    """
    if os.path.isfile(model_path):
        return model_path
    if not os.path.isdir(model_path):
        return model_path  # engine will surface the error
    if family in _DIR_SCAN_FAMILIES:
        return model_path

    if model_file:
        exact = os.path.join(model_path, model_file)
        if not os.path.isfile(exact):
            raise RuntimeError(
                f"{family}: requested model_file {model_file!r} is missing in "
                f"{model_path} — refusing to fall back to another weights file "
                f"(that would silently load the wrong variant/quant). Provision "
                f"it first (catalog required_files / download gate)."
            )
        logger.info("resolved %s → %s via model_file=%r", model_path, exact, model_file)
        return exact

    ggufs = sorted(_glob.glob(os.path.join(model_path, "*.gguf")))
    if len(ggufs) == 0:
        return model_path
    if len(ggufs) == 1:
        return ggufs[0]

    # Multi-GGUF: read the family's model_spec for canonical filenames
    spec: dict[str, Any] = {}
    spec_path = (
        Path(_AUDIOCPP_MODEL_SPECS) / f"{family}.json"
        if _AUDIOCPP_MODEL_SPECS else None
    )
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    canonical_names: set[str] = set()
    for pkg in spec.get("packages", []):
        strip_prefix = pkg.get(
            "strip_prefix",
            spec.get("package_defaults", {}).get("strip_prefix", ""),
        )
        for fname in pkg.get("files", []):
            name = fname
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix):].lstrip("/")
            canonical_names.add(os.path.basename(name))

    def _variant_match(needle: str, pool: list[str]) -> Optional[str]:
        nl = needle.lower()
        for g in pool:
            if nl in os.path.basename(g).lower():
                return g
        return None

    # 1. Variant match
    if variant:
        canonical_pool = [g for g in ggufs if os.path.basename(g) in canonical_names]
        hit = _variant_match(variant, canonical_pool) or _variant_match(variant, ggufs)
        if hit:
            logger.info("resolved %s → %s via variant=%r", model_path, hit, variant)
            return hit

    # 2. model_specs default package
    packages = spec.get("packages", [])
    if packages:
        pkg = next((p for p in packages if p.get("default")), packages[0])
        files = pkg.get("files", [])
        if files:
            strip_prefix = pkg.get(
                "strip_prefix",
                spec.get("package_defaults", {}).get("strip_prefix", ""),
            )
            fname = files[0]
            if strip_prefix and fname.startswith(strip_prefix):
                fname = fname[len(strip_prefix):].lstrip("/")
            resolved = os.path.join(model_path, fname)
            if os.path.isfile(resolved):
                return resolved

    # 3. Sidecar exclusion heuristic
    _SIDECAR_MARKERS = ("extras", "tokenizer", "vae", "predictor", "embed", "-lm-")
    main_candidates = [
        g for g in ggufs
        if not any(m in os.path.basename(g).lower() for m in _SIDECAR_MARKERS)
    ]
    if main_candidates:
        pick = max(main_candidates, key=lambda p: os.path.getsize(p))
        logger.info("resolved %s → %s via sidecar exclusion + size", model_path, pick)
        return pick
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# ManagedModel — the loaded model session (NATIVE, no subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class ManagedModel:
    """A loaded model session backed by the native audiocpp engine_runtime.

    The model is loaded into GPU memory via libaudiocore_native.so and stays
    resident until unload() is called. No subprocess, no HTTP — the engine
    runs in-process, exactly like every other ComfyUI model.

    VRAM accounting (AudiocoreLoadedModel) makes GPU usage visible to
    ComfyUI's scheduler so /free can clean up properly.
    """

    _active_model: ManagedModel | None = None

    def __init__(
        self,
        family: str,
        path: str,
        backend: str = "cuda",
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.family = family
        self.path = path
        self.backend = backend
        self._extras = dict(extras or {})
        self._variant = str(self._extras.get("variant", "") or "")
        self._session: NativeSession | None = None
        # Torch-pipeline engines (moss_sfx_v2 — TTS-Audio-Suite pattern).
        # Exactly one of _session / _torch_engine is set, decided by family
        # + checkpoint format in load().
        self._torch_engine: Any = None
        self._estimated_vram: int = _FAMILY_VRAM_ESTIMATE.get(
            family, 4 * 1024 * 1024 * 1024,
        )
        self._loaded_model_wrapper: AudiocoreLoadedModel | None = None

    def load(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
        **extras: Any,
    ) -> bool:
        """Load the model into GPU memory.

        moss_sfx_v2 runs the pure-torch diffusion pipeline (TTS-Audio-Suite
        pattern); every other family runs the native C++ session as before.
        """
        if extras:
            self._extras.update(extras)
            v = extras.get("variant")
            if v:
                self._variant = str(v)

        if self._torch_engine is not None:
            return True
        if self.family == "moss_sfx_v2":
            return self._load_torch(on_progress=on_progress)
        if self._session is not None:
            return True

        # Hot path: reuse active session if family + path + variant match
        active = ManagedModel._active_model
        if (
            active is not None
            and active is not self
            and active.family == self.family
            and active.path == self.path
            and active._variant == self._variant
            and active._session is not None
        ):
            self._session = active._session
            self._estimated_vram = active._estimated_vram
            logger.info(
                "reusing cached %s session (path=%s variant=%r)",
                self.family, self.path, self._variant,
            )
            return True

        # Cold path: destroy any previous session, create a new one
        prev = ManagedModel._active_model
        if prev is not None and prev is not self:
            logger.info("unloading %s before loading %s", prev.family, self.family)
            prev._cleanup()

        try:
            resolved_path = _resolve_model_file(
                self.family, self.path, variant=self._variant,
                model_file=self._extras.get("model_file", ""),
            )
            task = _FAMILY_TASK.get(self.family, "tts")
            # Variant-routed task (the 1.7B stroke): a stated variant
            # steers the C++ session task off the family default.
            variant_task = _VARIANT_TASK.get(self._variant.lower())
            if variant_task:
                logger.info(
                    "routing %s task %s → %s via variant=%r",
                    self.family, task, variant_task, self._variant,
                )
                task = variant_task
            model_spec = _discover_model_spec(self.family, resolved_path)

            self._session = NativeSession(
                registry=get_registry(),
                family=self.family,
                model_path=resolved_path,
                task=task,
                backend=self.backend,
                device=0,
                threads=4,
                model_spec_override=model_spec,
                variant=self._variant or None,
            )
            ManagedModel._active_model = self
            self._register_in_model_management()
            logger.info(
                "loaded %s from %s (variant=%r) natively",
                self.family, resolved_path, self._variant,
            )
            return True
        except (NativeError, Exception) as e:
            logger.error("load failed for %s: %s", self.family, e)
            self._session = None
            return False

    def _load_torch(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Load moss_sfx_v2 via the pure-torch diffusion pipeline.

        The HF checkpoint is identified by model_index.json. GGUF dirs are
        retired for this family — the torch pipeline cannot read them, and a
        silent fallback would just resurrect the crash class. Fail loud.
        """
        if not os.path.isfile(os.path.join(self.path, "model_index.json")):
            raise RuntimeError(
                "moss_sfx_v2 requires the HF torch checkpoint "
                "(a directory with model_index.json) — GGUF dirs are retired. "
                f"Use MOSS-SoundEffect-v2.0-src, got: {self.path}"
            )

        # Hot path: reuse the active torch engine (same family + path)
        active = ManagedModel._active_model
        if (
            active is not None
            and active is not self
            and active.family == self.family
            and active.path == self.path
            and active._torch_engine is not None
        ):
            self._torch_engine = active._torch_engine
            logger.info(
                "reusing cached %s torch engine (path=%s)",
                self.family, self.path,
            )
            return True

        # Cold path: evict any previous model, then load the pipeline
        prev = ManagedModel._active_model
        if prev is not None and prev is not self:
            logger.info("unloading %s before loading %s", prev.family, self.family)
            prev._cleanup()

        try:
            from .engines.moss_sfx_v2 import TorchSfxEngine

            engine = TorchSfxEngine(self.path)
            loaded = (
                engine.load()
                if on_progress is None
                else engine.load(on_progress=on_progress)
            )
            if not loaded:
                return False
            self._torch_engine = engine
            ManagedModel._active_model = self
            # No current_loaded_models registration: the pipeline is a plain
            # torch object — ComfyUI's gc + empty_cache manage its VRAM. The
            # AudiocoreLoadedModel protocol existed for the C++ session.
            logger.info("loaded %s from %s (torch pipeline)", self.family, self.path)
            return True
        except RuntimeError:
            raise
        except Exception as e:
            logger.error("torch load failed for %s: %s", self.family, e)
            self._torch_engine = None
            return False

    def _register_in_model_management(self) -> None:
        try:
            from comfy import model_management
        except ImportError:
            return
        if self._loaded_model_wrapper is not None:
            try:
                model_management.current_loaded_models.remove(self._loaded_model_wrapper)
            except ValueError:
                pass
        self._loaded_model_wrapper = AudiocoreLoadedModel(self)
        model_management.current_loaded_models.insert(0, self._loaded_model_wrapper)

    def _unregister_from_model_management(self) -> None:
        if self._loaded_model_wrapper is None:
            return
        try:
            from comfy import model_management
            model_management.current_loaded_models.remove(self._loaded_model_wrapper)
        except (ValueError, ImportError):
            pass
        self._loaded_model_wrapper = None

    def _free_session(self) -> None:
        """Destroy the native session WITHOUT touching current_loaded_models.

        Called from model_unload() during ComfyUI's free_memory iteration.
        """
        if self._session is not None:
            self._session.destroy()
            self._session = None
        if self._torch_engine is not None:
            self._torch_engine.unload()
            self._torch_engine = None
        if ManagedModel._active_model is self:
            ManagedModel._active_model = None
        self._loaded_model_wrapper = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _cleanup(self) -> None:
        """Full teardown: unregister + destroy session."""
        self._unregister_from_model_management()
        if self._session is not None:
            self._session.destroy()
            self._session = None
        if self._torch_engine is not None:
            self._torch_engine.unload()
            self._torch_engine = None
        if ManagedModel._active_model is self:
            ManagedModel._active_model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def unload(self) -> None:
        self._cleanup()

    # ─── Inference: speech (TTS + SFX) ─────────────────────────────────────

    def run_tts(self, text: str, **kwargs: Any) -> tuple[list[float], int]:
        """Run TTS or SFX inference. Returns (pcm_float32, sample_rate).

        ALL kwargs are forwarded to the engine — no filtering, no pydantic
        guard, no silent drops. guidance_scale, num_inference_steps,
        duration_seconds reach the engine for SFX. temperature, top_p, top_k
        reach the engine for TTS.
        """
        self._ensure_loaded()
        if self._torch_engine is not None:
            return self._run_torch_sfx(text, kwargs, None)
        request = self._build_speech_request(text, kwargs)
        assert self._session is not None
        pcm, sr, _channels = self._session.run(request)
        return pcm, sr

    def run_tts_streaming(
        self, text: str, *, on_progress: Optional[Callable] = None, **kwargs: Any,
    ) -> tuple[list[float], int]:
        """Run TTS/SFX with optional progress callback. Returns (pcm, sample_rate)."""
        self._ensure_loaded()
        if self._torch_engine is not None:
            return self._run_torch_sfx(text, kwargs, on_progress)
        assert self._session is not None
        if on_progress is not None:
            self._session.set_progress(
                lambda step, total, label: on_progress(step, total)
            )
        try:
            request = self._build_speech_request(text, kwargs)
            pcm, sr, _ch = self._session.run(request)
            return pcm, sr
        finally:
            if on_progress is not None:
                self._session.set_progress(None)

    # ─── Inference: SFX via torch pipeline ────────────────────────────────

    # Node inputs consumed by the v2 diffusion pipeline (node name → pipeline
    # kwarg). The shared AudiocoreTTS surface also carries TTS sampler/voice
    # inputs the diffusion pipeline does not have; those are listed in
    # _TORCH_SFX_IGNORED and LOGGED on every run — never silently dropped.
    _TORCH_SFX_MAP = {
        "seed": "seed",
        "guidance_scale": "guidance_scale",
        "num_inference_steps": "num_inference_steps",
        "duration_seconds": "duration_seconds",
    }
    _TORCH_SFX_IGNORED = frozenset({
        "mode", "voice", "language", "temperature", "top_p", "top_k",
        "speed", "repetition_penalty", "reference_audio", "reference_text",
        "speaker_name", "instruct", "voice_file", "voice_pca_strengths",
        # An embedding steers TTS voice cloning only; it cannot condition a
        # diffusion SFX pipeline. Reaching it here (node line 426-427) with a
        # voice_file set) is logged, never silently dropped.
        "speaker_embedding",
    })

    def _run_torch_sfx(
        self,
        text: str,
        kwargs: dict[str, Any],
        on_progress: Optional[Callable] = None,
    ) -> tuple[list[float], int]:
        """Run the torch SFX diffusion pipeline. Returns (pcm, sample_rate).

        No silent drops, two gates:
        1. The pydantic contract (schemas.build_speech_request) — undeclared
           params raise ParamDropError.
        2. The explicit mapping above — every node input is either mapped to
           a pipeline kwarg, logged as ignored-by-design (TTS-only inputs on
           a diffusion model), or RAISES.
        """
        # Gate 1: pydantic extra='forbid' — undeclared params HARD FAIL.
        _build_speech_request_validated(text, kwargs)
        # Gate 2: explicit mapping — unknown inputs HARD FAIL.
        assert self._torch_engine is not None
        unknown = (
            set(kwargs)
            - set(self._TORCH_SFX_MAP)
            - self._TORCH_SFX_IGNORED
        )
        if unknown:
            raise ParamDropError(
                "moss_sfx_v2 (torch pipeline): node input(s) have no mapping: "
                f"{sorted(unknown)}. Add them to _TORCH_SFX_MAP or "
                "_TORCH_SFX_IGNORED — never silently dropped."
            )
        ignored = sorted(k for k in self._TORCH_SFX_IGNORED if k in kwargs)
        if ignored:
            logger.info(
                "moss_sfx_v2 (torch): TTS-only inputs have no effect on the "
                "diffusion pipeline: %s",
                ", ".join(ignored),
            )
        gen: dict[str, Any] = {
            dst: kwargs[src] for src, dst in self._TORCH_SFX_MAP.items() if src in kwargs
        }
        pcm, sr = self._torch_engine.generate(
            str(text), on_progress=on_progress, **gen,
        )
        return pcm, sr

    # ─── Inference: music (ACE-Step) ───────────────────────────────────────

    def run_music(self, caption: str, **kwargs: Any) -> tuple[list[float], int, int]:
        """Run music generation. Returns (pcm, sample_rate, channels)."""
        self._ensure_loaded()
        request = self._build_music_request(caption, kwargs)
        assert self._session is not None
        pcm, sr, ch = self._session.run(request)
        return pcm, sr, ch

    def run_music_streaming(
        self, caption: str, *, on_progress: Optional[Callable] = None, **kwargs: Any,
    ) -> tuple[list[float], int, int]:
        """Run music generation with optional progress. Returns (pcm, sr, ch)."""
        self._ensure_loaded()
        assert self._session is not None
        if on_progress is not None:
            self._session.set_progress(
                lambda step, total, label: on_progress(step, total)
            )
        try:
            request = self._build_music_request(caption, kwargs)
            pcm, sr, ch = self._session.run(request)
            return pcm, sr, ch
        finally:
            if on_progress is not None:
                self._session.set_progress(None)

    # ─── Request builders ──────────────────────────────────────────────────
    #
    # These delegate to schemas.py — the declarative pydantic contract.
    # Every field is validated (extra='forbid'), every node name is translated
    # via an explicit map, and the routing (body root vs options sub-object)
    # is an explicit frozenset. No silent drops, ever. If a param isn't
    # declared in the pydantic model, it RAISES ParamDropError.

    @staticmethod
    def _build_speech_request(text: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build + validate the JSON request for TTS or SFX.

        Delegates to schemas.build_speech_request — the pydantic contract
        catches typos, misroutes, and undeclared params via extra='forbid'.
        Raises ParamDropError (a ValueError) on any violation.
        """
        return _build_speech_request_validated(text, kwargs)

    @staticmethod
    def _build_music_request(caption: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build + validate the JSON request for ACE-Step music.

        Delegates to schemas.build_music_request — the pydantic contract
        catches typos, misroutes, and undeclared params via extra='forbid'.
        Raises ParamDropError (a ValueError) on any violation.
        """
        return _build_music_request_validated(caption, kwargs)

    def compute_embedding(self, wav_path: str) -> list[float]:
        """Compute a speaker embedding via the native engine."""
        self._ensure_loaded()
        if self._torch_engine is not None:
            raise RuntimeError(
                "compute_embedding is a C++-engine feature — the torch SFX "
                "pipeline has no embedding extraction"
            )
        assert self._session is not None
        request: dict[str, Any] = {
            "input": "",
            "options": {"extract_embedding_only": True},
            "voice_ref": wav_path,
        }
        # The embedding comes back as a named audio output or text output;
        # for now this is a passthrough — the engine handles it.
        _pcm, _sr, _ch = self._session.run(request)
        return []

    def _ensure_loaded(self) -> None:
        if self._session is not None or self._torch_engine is not None:
            return
        active = ManagedModel._active_model
        if (
            active is not None
            and active.family == self.family
            and active.path == self.path
            and active._variant == self._variant
        ):
            if active._torch_engine is not None:
                self._torch_engine = active._torch_engine
                logger.info(
                    "auto-recovered %s torch engine from active cache",
                    self.family,
                )
                return
            if active._session is not None:
                self._session = active._session
                self._estimated_vram = active._estimated_vram
                logger.info("auto-recovered %s session from active cache", self.family)
                return
        logger.info("model was evicted — cold-reloading %s", self.family)
        if not self.load():
            raise RuntimeError(
                f"model not loaded and auto-reload failed for {self.family}"
            )

    @property
    def loaded(self) -> bool:
        return self._session is not None or self._torch_engine is not None


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI model_management integration
# ─────────────────────────────────────────────────────────────────────────────

class _AudiocorePatcherStub:
    """Minimal stub that quacks like comfy.model_patcher.ModelPatcher."""

    __slots__ = ("_managed",)

    def __init__(self, managed: ManagedModel):
        self._managed = managed

    @property
    def model(self):
        return self

    def is_dynamic(self) -> bool:
        return False

    def model_size(self) -> int:
        return self._managed._estimated_vram

    def loaded_size(self) -> int:
        if self._managed._session is not None:
            return self._managed._estimated_vram
        return 0

    def current_loaded_device(self):
        import torch
        if self._managed._session is not None and torch.cuda.is_available():
            return torch.device("cuda", 0)
        return None

    def partially_unload_ram(self, ram_to_unload: int) -> int:
        return 0

    def model_mmap_residency(self, free: bool = False):
        total = self._managed._estimated_vram
        if self._managed._session is not None:
            return (total, total)
        return (0, total)

    def pinned_memory_size(self) -> int:
        return 0


class AudiocoreLoadedModel:
    """LoadedModel-shaped wrapper for native audiocore sessions.

    Registers in ComfyUI's current_loaded_models so the scheduler can account
    for our VRAM usage and call model_unload() during /free.
    """

    def __init__(self, managed: ManagedModel):
        self._managed = managed
        try:
            import torch
            self.device = torch.device("cuda", 0) if torch.cuda.is_available() else None
        except ImportError:
            self.device = None
        self.currently_used = True
        self._patcher = _AudiocorePatcherStub(managed)

    @property
    def model(self):
        return self._patcher

    def model_memory(self) -> int:
        return self._managed._estimated_vram

    def model_loaded_memory(self) -> int:
        return self.model_memory()

    def model_offloaded_memory(self) -> int:
        return 0

    def model_memory_required(self, device) -> int:
        return self.model_memory()

    def model_mmap_residency(self, free: bool = False):
        return self._patcher.model_mmap_residency(free=free)

    def model_load(self, lowvram_model_memory: int = 0, force_patch_weights: bool = False):
        return self

    def should_reload_model(self, force_patch_weights: bool = False) -> bool:
        return False

    def model_unload(self, memory_to_free: int | None = None, unpatch_weights: bool = True) -> bool:
        if memory_to_free is not None and memory_to_free < self.model_memory():
            return False
        self._managed._free_session()
        return True

    def model_use_more_vram(self, extra_memory: int, force_patch_weights: bool = False) -> None:
        pass

    def __eq__(self, other):
        return isinstance(other, AudiocoreLoadedModel) and self._managed is other._managed

    def __hash__(self):
        return id(self._managed)

    def is_dead(self) -> bool:
        return self._managed._session is None

    def real_model(self):
        # Return a dummy callable instead of None. ComfyUI's cleanup_models()
        # calls real_model() on every loaded model during GC; the old code
        # returned None here, causing TypeError: 'NoneType' is not callable
        # when the subprocess was dead. With native sessions, the session
        # is either alive or destroyed — this stub avoids the crash entirely.
        return _noop_real_model

    # ComfyUI also checks for `.model` to have a `real_model` callable in
    # some code paths — provide it on the stub too.


def _noop_real_model(*args: Any, **kwargs: Any) -> Any:
    """No-op callable returned by AudiocoreLoadedModel.real_model().

    Prevents TypeError when ComfyUI's cleanup_models() calls real_model()
    on our wrapper. The native session manages its own lifecycle.
    """
    return None
