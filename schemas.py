"""Declarative pydantic contracts for the native engine request surface.

EVERY field the C++ engine reads is declared here. extra='forbid' catches
typos, misroutes, and undocumented params — NO SILENT DROPS, ever.

Field names are the EXACT names that appear in the JSON body sent to
libaudiocore_native.so via audiocore_session_run(). They are verified against
the C++ engine's option reads:

    moss_sfx_v2:  src/models/moss/moss_sfx_v2/session.cpp
    qwen3_tts:    src/models/qwen3_tts/session.cpp
    ace_step:     src/models/ace_step/request_parser.cpp
    native_api:   app/native_api.cpp (C ABI: body → TaskRequest)

ARCHITECTURE:
  1. Node kwargs → translated via _SPEECH_PARAM_MAP / _MUSIC_PARAM_MAP
     (node param names ≠ engine field names — the map IS the single source
     of truth for the translation).
  2. Translated dict → validated by the pydantic model. extra='forbid'
     RAISES on any undeclared param. A typo can never vanish silently.
  3. Validated dict → split into body-root + 'options' sub-object.
     _SPEECH_OPTIONS_ONLY / _MUSIC_OPTIONS_ONLY is the explicit routing
     table — these fields MUST go via the options merge because
     native_api.cpp does NOT read them from body root.
  4. The final dict crosses the ctypes boundary.

The split is the ONLY place routing happens — if native_api.cpp changes
(adds/removes a body-root field read), these sets must change too. The
cross-contract test (test_audiocore_param_contract.py) catches the drift.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Node param name → engine field name translation
# ─────────────────────────────────────────────────────────────────────────────
# The ONLY params whose name changes between the ComfyUI node surface and the
# engine's JSON body. Everything else passes through under its engine name.

_SPEECH_PARAM_MAP: dict[str, str] = {
    "instruct": "instructions",         # node → body root → options["instruct"]
    "reference_audio": "voice_ref",     # node → body root → VoiceReference.audio
    "speaker_name": "speaker",          # node → options["speaker"] (qwen3_tts:615)
}

_MUSIC_PARAM_MAP: dict[str, str] = {
    "duration": "duration_seconds",
    "n_diffusion_steps": "num_inference_steps",
    "temperature": "lm_temperature",
    "top_p": "lm_top_p",
}


# ─────────────────────────────────────────────────────────────────────────────
# Routing: which fields go in the 'options' sub-object vs body root
# ─────────────────────────────────────────────────────────────────────────────
# native_api.cpp explicitly reads these via add_option_from_json(body root):
#   seed, temperature, top_k, top_p, max_tokens, max_steps, repetition_penalty,
#   guidance_scale, num_inference_steps, do_sample, num_beams, duration_seconds,
#   text_chunk_size
# And reads these via special body-root handlers:
#   input, language, instructions, voice, voice_ref, reference_text
#
# Fields NOT in either list MUST go via the options sub-object merge:

_SPEECH_OPTIONS_ONLY: frozenset[str] = frozenset({
    "speed",            # only omnivoice reads options["speed"] (session.cpp:86)
    "speaker",          # qwen3_tts reads options["speaker"] (session.cpp:615)
    "negative_prompt",  # moss_sfx_v2 reads options["negative_prompt"] (session.cpp:460)
})

_MUSIC_OPTIONS_ONLY: frozenset[str] = frozenset({
    "lyrics", "duration_seconds", "lm_temperature", "lm_top_p",
    "lm_cfg_scale", "max_tokens", "negative_prompt", "instruction",
    "bpm", "keyscale",
})


# ─────────────────────────────────────────────────────────────────────────────
# Speech request (TTS + SFX families)
# ─────────────────────────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    """Declarative contract for the speech/SFX request.

    Field names ARE the engine's JSON field names. extra='forbid' catches
    any param not declared here — no silent drops, ever. A typo raises
    ParamDropError instead of vanishing into the engine.

    Verified against:
      - native_api.cpp: body.find() + add_option_from_json() field list
      - moss_sfx_v2/session.cpp: find_option / parse_*_option calls
      - qwen3_tts/session.cpp: find_option / parse_*_option calls
    """

    model_config = ConfigDict(extra="forbid")

    # Required: the prompt/text
    input: str = Field(description="Text/prompt → TaskRequest.text_input")

    # Body-root fields (native_api.cpp reads from body root)
    language: Optional[str] = Field(default=None, description="→ Transcript.language")
    instructions: Optional[str] = Field(default=None, description="→ options['instruct']")
    voice: Optional[str] = Field(default=None, description="Named voice → VoiceReference.cached_voice_id")
    voice_ref: Optional[str] = Field(default=None, description="WAV path → VoiceReference.audio")
    reference_text: Optional[str] = Field(default=None, description="→ options['reference_text']")

    # Sampling params (body root → add_option_from_json → request.options)
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295,
        description="Random seed (moss_sfx_v2:470, qwen3_tts:79)")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0,
        description="Sampling temperature (qwen3_tts:50)")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0,
        description="Nucleus sampling p (qwen3_tts:56)")
    top_k: Optional[int] = Field(default=None, ge=0, le=1000,
        description="Top-k sampling (qwen3_tts:53)")
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0, le=10.0,
        description="Repetition penalty (qwen3_tts:59)")
    max_tokens: Optional[int] = Field(default=None, ge=1,
        description="Max output tokens (qwen3_tts:36)")
    max_steps: Optional[int] = Field(default=None, ge=1,
        description="Max inference steps")
    do_sample: Optional[bool] = Field(default=None,
        description="Enable sampling (qwen3_tts:42)")
    num_beams: Optional[int] = Field(default=None, ge=1,
        description="Beam search width")
    text_chunk_size: Optional[int] = Field(default=None, ge=1,
        description="Text chunk size (qwen3_tts:362)")

    # Diffusion params (body root → add_option_from_json → request.options)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=50.0,
        description="CFG guidance (moss_sfx_v2:468)")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=1000,
        description="Diffusion steps (moss_sfx_v2:466)")
    duration_seconds: Optional[float] = Field(default=None, gt=0.0, le=600.0,
        description="Output duration seconds (moss_sfx_v2:463)")

    # Options-only fields (native_api.cpp does NOT read from body root)
    speed: Optional[float] = Field(default=None, ge=0.1, le=10.0,
        description="Speed — ONLY omnivoice reads this; others IGNORE it")
    speaker: Optional[str] = Field(default=None,
        description="Named speaker ID (qwen3_tts:615 reads options['speaker'])")
    negative_prompt: Optional[str] = Field(default=None,
        description="Negative prompt (moss_sfx_v2:460 reads options['negative_prompt'])")


# ─────────────────────────────────────────────────────────────────────────────
# Music request (ACE-Step)
# ─────────────────────────────────────────────────────────────────────────────

class MusicRequest(BaseModel):
    """Declarative contract for the music request (ACE-Step).

    Verified against ace_step/request_parser.cpp option reads.
    """

    model_config = ConfigDict(extra="forbid")

    input: str = Field(description="Caption/prompt")

    # Body root (native_api.cpp reads via add_option_from_json)
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295,
        description="Random seed")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=50.0,
        description="Diffusion guidance scale")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=1000,
        description="Diffusion steps")

    # Options-only (engine reads from request.options via request_parser.cpp)
    lyrics: Optional[str] = Field(default=None, description="Lyrics (request_parser.cpp:126)")
    duration_seconds: Optional[float] = Field(default=None, gt=0.0, le=600.0,
        description="Duration seconds (request_parser.cpp:187)")
    lm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0,
        description="LM temperature")
    lm_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0,
        description="LM nucleus sampling p")
    lm_cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=50.0,
        description="LM CFG scale")
    max_tokens: Optional[int] = Field(default=None, ge=1,
        description="Max output tokens (safety cap)")
    negative_prompt: Optional[str] = Field(default=None,
        description="Negative prompt (request_parser.cpp:129)")
    instruction: Optional[str] = Field(default=None,
        description="Instruction (request_parser.cpp:132)")
    bpm: Optional[int] = Field(default=None, description="Beats per minute")
    keyscale: Optional[int] = Field(default=None, description="Key scale")


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────

class ParamDropError(ValueError):
    """Raised when a parameter would be silently dropped.

    This is the HARD FAIL for the 'no silent drops' invariant. If a param
    can't be validated, mapped, or forwarded, this fires — never a silent
    swallow.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Validation + serialization entry points
# ─────────────────────────────────────────────────────────────────────────────

def build_speech_request(text: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate node kwargs, validate, and serialize the speech request.

    1. Strip node-only metadata (mode, speaker_embedding, voice_file,
       voice_pca_strengths) — these are handled by the NODE, not the engine.
    2. Translate node param names → engine field names via _SPEECH_PARAM_MAP.
    3. Validate via SpeechRequest (extra='forbid' catches misroutes).
    4. Split into body-root + options sub-object.
    """
    # Node-only metadata — consumed by AudiocoreTTS.synthesize(), NOT forwarded
    _NODE_ONLY = frozenset({
        "mode",                 # tts/clone/design — node infers engine behavior
        "speaker_embedding",    # .voice files — not yet wired via C ABI
        "voice_file",           # consumed by node (binary read)
        "voice_pca_strengths",  # consumed by node (PCA steering)
    })

    translated: dict[str, Any] = {"input": text}

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if key in _NODE_ONLY:
            if key == "speaker_embedding":
                raise ParamDropError(
                    "speaker_embedding (from .voice files) is not yet wired "
                    "through the native C ABI. The engine has no path for raw "
                    "float vectors via audiocore_session_run. Use voice_ref "
                    "(a WAV file) for voice cloning instead. This is a HARD "
                    "FAIL — the old HTTP server silently dropped this param too."
                )
            continue  # other node-only metadata: silently consumed by the node

        engine_name = _SPEECH_PARAM_MAP.get(key, key)
        if engine_name == "language":
            translated[engine_name] = _normalize_language(value)
        else:
            translated[engine_name] = value

    # Validate — extra='forbid' catches ANY undeclared param
    try:
        model = SpeechRequest(**translated)
    except ValidationError as e:
        raise ParamDropError(
            f"Speech request validation failed — a param was misrouted, "
            f"misspelled, or undeclared:\n{e}\n\n"
            f"Input dict: {translated}\n"
            f"This is a HARD FAIL: add the field to SpeechRequest in "
            f"schemas.py or fix the typo."
        ) from e

    return _split_speech(model.model_dump(exclude_none=True, mode="json"))


def build_music_request(caption: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate node kwargs, validate, and serialize the music request."""
    translated: dict[str, Any] = {"input": caption}

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        engine_name = _MUSIC_PARAM_MAP.get(key, key)
        translated[engine_name] = value

    # Legacy semantics (2026-08-11): the old music card's SCHEMA declared
    # ``n_diffusion_steps`` default 0 with tooltip "0 = engine default".
    # Saved pipes authored under that contract still send 0 — but the
    # engine contract (MusicRequest.num_inference_steps) is ge=1, so a raw
    # 0 would HARD FAIL the request instead of running. 0 here means
    # "unset" — drop it and let the engine apply its own default (8).
    # Positive values pass through unchanged (pinned by
    # test_no_silent_param_drops). This is a deliberate semantic
    # translation, not a silent drop: the request is still validated
    # against MusicRequest afterwards.
    steps = translated.get("num_inference_steps")
    if steps is not None and steps <= 0:
        del translated["num_inference_steps"]

    # Duration safety cap (50 tokens/sec)
    duration = translated.get("duration_seconds")
    if duration is not None:
        translated.setdefault("max_tokens", int(float(duration) * 50))

    try:
        model = MusicRequest(**translated)
    except ValidationError as e:
        raise ParamDropError(
            f"Music request validation failed — a param was misrouted, "
            f"misspelled, or undeclared:\n{e}\n\n"
            f"Input dict: {translated}\n"
            f"This is a HARD FAIL: add the field to MusicRequest in "
            f"schemas.py or fix the typo."
        ) from e

    return _split_music(model.model_dump(exclude_none=True, mode="json"))


# ─── Internal helpers ─────────────────────────────────────────────────────

def _normalize_language(lang: str) -> str:
    """Normalize ISO 639-1 → full lowercase name (engine convention)."""
    _LANGUAGE_FULL_NAME: dict[str, str] = {
        "en": "english", "eng": "english",
        "zh": "chinese", "cmn": "chinese",
        "zh-cn": "chinese", "zh-tw": "chinese", "zh-hans": "chinese",
        "ja": "japanese", "jpn": "japanese",
        "ko": "korean", "kor": "korean",
    }
    if not lang:
        return ""
    return _LANGUAGE_FULL_NAME.get(lang.strip().lower(), lang)


def _split_speech(flat: dict[str, Any]) -> dict[str, Any]:
    """Split flat validated dict into body-root + options sub-object."""
    body: dict[str, Any] = {}
    options: dict[str, Any] = {}
    for key, value in flat.items():
        if key in _SPEECH_OPTIONS_ONLY:
            options[key] = value
        else:
            body[key] = value
    if options:
        body["options"] = options
    return body


def _split_music(flat: dict[str, Any]) -> dict[str, Any]:
    """Split flat validated dict into body-root + options sub-object."""
    body: dict[str, Any] = {}
    options: dict[str, Any] = {}
    for key, value in flat.items():
        if key in _MUSIC_OPTIONS_ONLY:
            options[key] = value
        else:
            body[key] = value
    if options:
        body["options"] = options
    return body
