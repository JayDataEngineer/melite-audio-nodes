"""qwen3_tts — text-to-speech via the upstream qwen-tts Python package.

Proxies the official `qwen_tts` package (pip-installed) which wraps
Qwen3TTSForConditionalGeneration + Qwen3TTSProcessor. No GGUF, no
conversion. The model is loaded directly from the HF source dir
(config.json + model.safetensors + speech_tokenizer/ + tokenizer bits).

Three call modes match the audiocore surface:
  - tts         -> generate_custom_voice(text, speaker, language)
  - voice_clone -> generate_voice_clone(text, ref_audio, ref_text)
                   (needs a "base" model variant, not "custom_voice")
  - design      -> generate_voice_design(text, instruct)
                   (needs a "voicedesign" model variant)

Hyperparameters (temperature, top_p, top_k, repetition_penalty, seed)
are forwarded into the underlying transformers generate() call.

Voice Studio extensions (export / preview / load_qvoice) implement the
.qvoice file format (lite + WDELTA) and the combo pipeline. See
``qvoice.py`` for the file format spec.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from ..core import resolve_model_folder

logger = logging.getLogger("audiocore.qwen3_tts")


# Default hyperparameters — match the audiocore node defaults where possible.
_DEFAULTS = {
    "speaker": "vivian",
    "language": "english",
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
}


class Qwen3TtsEngine:
    """Wraps qwen_tts.Qwen3TTSModel to match audiocore's session contract."""

    def __init__(self) -> None:
        self.model: Any = None          # qwen_tts.Qwen3TTSModel
        self._model_dir: str | None = None
        # Variant currently loaded: "customvoice" | "base" | "voicedesign".
        # Tracked so Voice Studio preview can decide whether the active
        # model is the right variant for the qvoice kind being previewed
        # (lite → base, wdelta → customvoice-after-patch).
        self._variant: str = "customvoice"

    # ── audiocore session contract ──────────────────────────────────────

    def load(self, path: str, **extras: Any) -> bool:
        """Load Qwen3TTSModel from `path` (HF source dir with config.json).

        Variant selection: the qwen_tts package gates features per model
        variant. The default load picks CustomVoice (plain TTS with
        predefined speakers). Pass `variant='base'` via extras for voice
        cloning, or `variant='voicedesign'` for instructed voice design.

        extras (JSON object from the LoadAudiocoreModel node):
          - variant:           'customvoice' (default) | 'base' | 'voicedesign'
          - speaker_encoder_path: ignored (the Base HF dir ships its own)
        """
        import contextlib
        import io

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

        # The qwen_tts package prints a flash-attn warning to stdout on
        # import — silence it so it doesn't pollute ComfyUI logs.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError as e:
                raise RuntimeError(
                    f"qwen3_tts: qwen-tts Python package not installed: {e}"
                ) from e

        import torch

        logger.info("qwen3_tts: loading Qwen3TTSModel from %s", hf_dir)
        self.model = Qwen3TTSModel.from_pretrained(
            hf_dir,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self._model_dir = hf_dir
        logger.info(
            "qwen3_tts: model ready (device=%s, speakers=%s)",
            self.model.device,
            (self.model.get_supported_speakers() or [])[:5],
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

        # Pop audiocore control kwargs that don't map to qwen_tts.
        mode = str(kwargs.get("mode", "tts")).lower()
        # `voice` is audiocore's speaker alias.
        speaker = str(kwargs.get("voice", kwargs.get("speaker", _DEFAULTS["speaker"])))
        language = self._coerce_language(kwargs.get("language", _DEFAULTS["language"]))
        instruct = kwargs.get("instruct", "") or ""
        seed = self._coerce_seed(kwargs.get("seed", 0))

        # Build gen kwargs forwarded to transformers generate().
        gen_kwargs: dict[str, Any] = {}
        for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if k in kwargs and kwargs[k] is not None and kwargs[k] != "":
                try:
                    gen_kwargs[k] = float(kwargs[k]) if k != "top_k" else int(kwargs[k])
                except (TypeError, ValueError):
                    pass
        if seed > 0:
            import torch
            torch.manual_seed(seed)
            gen_kwargs["seed"] = seed

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

        Uses qwen_tts' create_voice_clone_prompt to extract the x-vector
        embedding from the reference audio. Returns it as a flat list so
        the AudiocoreVoiceEmbedding node can cache + reuse it.
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
        prompt_items = self.model.create_voice_clone_prompt(
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
            try:
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

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

        voices_dir = voices_dir or _qv.VOICE_DIR_DEFAULT
        os.makedirs(voices_dir, exist_ok=True)
        unique_name = _qv.next_available_name(name, voices_dir)
        out_path = os.path.join(voices_dir, unique_name + ".qvoice")

        synth_text = sample_text or (
            "Hello, I am a friendly voice assistant. "
            "How can I help you today?"
        )
        if not instruct.strip():
            raise RuntimeError(
                "export_voice: instruct is required (VoiceDesign needs a "
                "style description, e.g. 'A warm female voice with a slight "
                "British accent, conversational pace.')"
            )

        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, seed=seed,
        )

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
            items = self.model.create_voice_clone_prompt(
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

        # Reconstruct VoiceClonePromptItem list. Apply voice_strength by
        # scaling the embedding's L2 distance from zero (uniform magnitude
        # scaling — preserves direction, modulates identity intensity).
        items: list[Any] = []
        for raw in items_raw:
            emb = raw.get("ref_spk_embedding")
            if emb is not None and strength != 1.0:
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                emb = emb * strength
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
            items = self.model.create_voice_clone_prompt(
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
            gen["emotion"] = emotion

        if items:
            # Apply voice_strength to the embedding tensor.
            if strength != 1.0:
                import torch
                for it in items:
                    emb = getattr(it, "ref_spk_embedding", None)
                    if isinstance(emb, torch.Tensor):
                        it.ref_spk_embedding = emb * strength
            audios, sr = self.model.generate_voice_clone(
                text=text, language=lang,
                voice_clone_prompt=items, **gen,
            )
        else:
            # No ref audio — use custom_voice with the patched talker.
            # instruct + emotion still apply; the patched weights carry
            # the voice identity from VoiceDesign.
            audios, sr = self.model.generate_custom_voice(
                text=text, speaker="vivian", language=lang, **gen,
            )

        # Speed is applied as a post-processing step (qwen_tts doesn't
        # expose it in generate()). We do a no-op pass for speed == 1.0.
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
        The final render uses qwen_tts' native rate parameter.
        """
        if not pcm or abs(speed - 1.0) < 0.001:
            return pcm
        n_out = max(1, int(len(pcm) / speed))
        idx = [int(i * speed) for i in range(n_out)]
        return [pcm[min(i, len(pcm) - 1)] for i in idx]

    def _ensure_variant(self, variant: str) -> None:
        """Load `variant` if it isn't already active. Bypasses ManagedModel.

        Uses the same _resolve_model_dir helper so HF cache lookup,
        flat-dir detection, and snapshot_download fallback all work.
        """
        if self.model is not None and self._variant == variant:
            return
        # Need to swap — drop current and load new.
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
        import contextlib
        import io
        import torch
        with contextlib.redirect_stdout(io.StringIO()):
            from qwen_tts import Qwen3TTSModel
        logger.info("qwen3_tts: loading variant=%s from %s", variant, hf_dir)
        self.model = Qwen3TTSModel.from_pretrained(
            hf_dir, dtype=torch.bfloat16, device_map="cuda:0",
        )
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

        qwen_tts.Qwen3TTSModel wraps Qwen3TTSForConditionalGeneration
        which exposes ``.talker`` (the LM) and ``.codec`` (the audio
        decoder). Different package versions name these differently, so
        we probe common paths.
        """
        m = self.model
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
    def _build_gen_kwargs(
        *, temperature: float, top_p: float, top_k: int,
        repetition_penalty: float, seed: int,
    ) -> dict[str, Any]:
        """Build the gen kwargs forwarded to qwen_tts.generate_*."""
        out: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
        }
        if int(seed) > 0:
            import torch
            torch.manual_seed(int(seed))
            out["seed"] = int(seed)
        return out

    # Alias so ManagedModel lifecycle (which calls .close() on the C++
    # session) works uniformly on either kind of _session.
    def close(self) -> None:
        self.unload()

    # ── dispatch helpers ────────────────────────────────────────────────

    def _custom_voice(self, text: str, speaker: str, language: str,
                      instruct: str, gen_kwargs: dict[str, Any]
                      ) -> tuple[list[float], int]:
        # The 0.6B CustomVoice model ignores instruct.
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
        """Convert qwen_tts' List[np.ndarray] → (list[float], int)."""
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
        """Map short codes → full names expected by qwen_tts.

        qwen_tts accepts: auto, chinese, english, french, german, italian,
        japanese, korean, portuguese, russian, spanish. Map common short
        forms; pass through unknown values (qwen_tts will validate).
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

        The qwen_tts package gates features per variant:
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
            resolve_model_folder("audiocpp", "AUDIOCPP_MODELS_DIR"),
            resolve_model_folder("qwen3_tts", "QWEN3_HF_ROOT"),
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
