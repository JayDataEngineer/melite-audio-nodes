"""qvoice — file format for Qwen3-TTS voice artifacts.

Two variants share the .qvoice extension; both are first-class inputs to
the audiocore voice pipeline:

  lite   (~5-25 MB) : serialized VoiceClonePromptItem list. Contains the
                     ref_code codec tokens + ref_spk_embedding extracted
                     from a VoiceDesign-generated reference clip. Used at
                     inference time for codec-level ICL cloning
                     (``--icl-only`` in the upstream CLI).

  wdelta (~1-3 GB)  : patched talker state_dict. The CustomVoice talker's
                     text_proj + token_embd tensors are overwritten with
                     Base's versions so the resulting model accepts a
                     continuous ECAPA embedding at the speaker slot while
                     retaining CV's instruct-tuned transformer norms.
                     Enables the "combo pipeline" (speaker embedding +
                     instruct simultaneously) AND supports voice_strength
                     scaling. Equivalent to ``--target-cv`` in the
                     upstream CLI.

Header (32 bytes, little-endian):
  offset 0   16 bytes  magic   — b"QVOICE-LITE\\x00\\x00\\x00\\x00\\x00"
                              or b"QVOICE-WDELTA\\x00\\x00\\x00"
  offset 16   4 bytes  version (u32) = 1
  offset 20   4 bytes  flags   (u32) — reserved, must be 0
  offset 24   8 bytes  payload_size (u64) — bytes following the header
  offset 32   ...      payload (torch.save output)

Both payloads are written via ``torch.save`` so tensor storage is
memory-mapped on read and round-trips losslessly.
"""
from __future__ import annotations

import io
import os
import struct
import time
from typing import Any, Literal

VOICE_DIR_DEFAULT = "/mnt/data/models/audio/voices"

MAGIC_LITE: bytes = b"QVOICE-LITE\x00\x00\x00\x00\x00"
MAGIC_WDELTA: bytes = b"QVOICE-WDELTA\x00\x00\x00"
assert len(MAGIC_LITE) == 16
assert len(MAGIC_WDELTA) == 16

VERSION: int = 1
HEADER_FMT: str = "<16sIIQ"
HEADER_SIZE: int = struct.calcsize(HEADER_FMT)  # 32

Kind = Literal["lite", "wdelta"]


# ── Magic / kind helpers ─────────────────────────────────────────────────


def _kind_for_magic(magic: bytes) -> Kind:
    if magic == MAGIC_LITE:
        return "lite"
    if magic == MAGIC_WDELTA:
        return "wdelta"
    raise ValueError(f"not a QVOICE file (magic={bytes(magic)!r})")


def _magic_for_kind(kind: Kind) -> bytes:
    if kind == "lite":
        return MAGIC_LITE
    if kind == "wdelta":
        return MAGIC_WDELTA
    raise ValueError(f"unknown qvoice kind: {kind!r}")


def detect_kind(path: str) -> Kind | None:
    """Read the 16-byte magic and return 'lite' / 'wdelta' / None.

    Returns None for missing files or non-qvoice files. Safe to call on
    arbitrary user input.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(16)
    except OSError:
        return None
    if len(magic) < 16:
        return None
    if magic == MAGIC_LITE:
        return "lite"
    if magic == MAGIC_WDELTA:
        return "wdelta"
    return None


# ── Writers ──────────────────────────────────────────────────────────────


def write_lite(
    path: str,
    *,
    name: str,
    instruct: str,
    sample_text: str,
    items: list[Any],
    voices_dir: str | None = None,
) -> int:
    """Write a lite qvoice from a list of qwen_tts.VoiceClonePromptItem.

    Returns the payload size in bytes (excluding the 32-byte header).

    The payload is a torch.save'd dict containing the per-item tensors
    (ref_code, ref_spk_embedding) plus metadata (name, instruct,
    sample_text, created_at). Items are stored as plain dicts so the
    loader does NOT depend on the qwen_tts package's dataclass layout
    at load time — only at write time.
    """
    import torch

    payload_obj: dict[str, Any] = {
        "format": "qvoice-lite",
        "version": VERSION,
        "name": name,
        "instruct": instruct,
        "sample_text": sample_text,
        "created_at": time.time(),
        "items": [
            {
                "ref_code": getattr(it, "ref_code", None),
                "ref_spk_embedding": getattr(it, "ref_spk_embedding", None),
                "x_vector_only_mode": bool(getattr(it, "x_vector_only_mode", False)),
                "icl_mode": bool(getattr(it, "icl_mode", False)),
                "ref_text": getattr(it, "ref_text", None),
            }
            for it in items
        ],
    }
    buf = io.BytesIO()
    torch.save(payload_obj, buf)
    payload = buf.getvalue()

    flags = 0  # reserved
    header = struct.pack(HEADER_FMT, MAGIC_LITE, VERSION, flags, len(payload))

    os.makedirs(os.path.dirname(path) or "." if voices_dir is None else voices_dir,
                exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return len(payload)


def write_wdelta(
    path: str,
    *,
    name: str,
    instruct: str,
    talker_state: dict[str, Any],
    text_proj_state: dict[str, Any] | None = None,
    token_embd_state: dict[str, Any] | None = None,
    source_ref_audio: str = "",
    sample_text: str = "",
) -> int:
    """Write a WDELTA qvoice from a patched talker state_dict.

    ``talker_state`` should be the full state_dict of the CV talker AFTER
    Base's text_proj + token_embd have been applied (see
    ``Qwen3TtsEngine._patch_cv_with_base``). The full state is stored so
    the loader can apply it with one ``load_state_dict`` call.

    Returns the payload size in bytes.
    """
    import torch

    payload_obj: dict[str, Any] = {
        "format": "qvoice-wdelta",
        "version": VERSION,
        "name": name,
        "instruct": instruct,
        "sample_text": sample_text,
        "source_ref_audio": source_ref_audio,
        "created_at": time.time(),
        "talker_state": talker_state,
        # Stored separately for forensics / partial-patch use. The full
        # talker_state above already includes these tensors; the duplicates
        # are a convenience for callers that want to apply ONLY the patch
        # tensors without overwriting the rest of the talker.
        "text_proj_state": text_proj_state or {},
        "token_embd_state": token_embd_state or {},
    }
    buf = io.BytesIO()
    torch.save(payload_obj, buf)
    payload = buf.getvalue()

    flags = 0
    header = struct.pack(HEADER_FMT, MAGIC_WDELTA, VERSION, flags, len(payload))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return len(payload)


# ── Readers ──────────────────────────────────────────────────────────────


def read_header(path: str) -> dict[str, Any]:
    """Read the 32-byte header — kind, version, flags, payload_size.

    Raises ValueError if the file is too small or the magic doesn't match.
    """
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"file too small to be a qvoice: {path}")
    magic, version, flags, payload_size = struct.unpack(HEADER_FMT, raw)
    return {
        "kind": _kind_for_magic(magic),
        "version": version,
        "flags": flags,
        "payload_size": payload_size,
    }


def read_lite(path: str) -> dict[str, Any]:
    """Read a lite qvoice — returns the payload dict with ``items``."""
    import torch

    h = read_header(path)
    if h["kind"] != "lite":
        raise ValueError(f"{path} is not a lite qvoice (kind={h['kind']})")
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        payload = f.read(h["payload_size"])
    return torch.load(io.BytesIO(payload), weights_only=False)


def read_wdelta(path: str) -> dict[str, Any]:
    """Read a WDELTA qvoice — returns the payload dict with state dicts."""
    import torch

    h = read_header(path)
    if h["kind"] != "wdelta":
        raise ValueError(f"{path} is not a wdelta qvoice (kind={h['kind']})")
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        payload = f.read(h["payload_size"])
    return torch.load(io.BytesIO(payload), weights_only=False)


def read(path: str) -> dict[str, Any]:
    """Read either kind — dispatches on the magic."""
    kind = detect_kind(path)
    if kind is None:
        # Give a more useful error than the raw ValueError from read_header.
        if os.path.isfile(path):
            raise ValueError(f"{path} exists but is not a qvoice (bad magic)")
        raise FileNotFoundError(path)
    return read_lite(path) if kind == "lite" else read_wdelta(path)


# ── Filesystem helpers ───────────────────────────────────────────────────


def list_voices(voices_dir: str = VOICE_DIR_DEFAULT) -> list[dict[str, Any]]:
    """Enumerate .qvoice files in ``voices_dir``.

    Returns a list of dicts with name, filename, path, size, kind,
    created_at, sorted by name.
    """
    if not os.path.isdir(voices_dir):
        return []
    out: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(voices_dir)):
        if not entry.endswith(".qvoice"):
            continue
        full = os.path.join(voices_dir, entry)
        try:
            st = os.stat(full)
        except OSError:
            continue
        kind = detect_kind(full) or "unknown"
        out.append({
            "name": entry[: -len(".qvoice")],
            "filename": entry,
            "path": full,
            "size": st.st_size,
            "kind": kind,
            "created_at": st.st_mtime,
        })
    return out


def resolve_voice_path(
    name_or_path: str,
    voices_dir: str = VOICE_DIR_DEFAULT,
) -> str:
    """Resolve a user-supplied voice reference to an absolute .qvoice path.

    Accepts:
      - Absolute path to a .qvoice file
      - Bare voice name ("Cherry") → looks up <voices_dir>/Cherry.qvoice
      - Name with extension ("Cherry.qvoice") → same

    Rejects any name containing path separators or ``..`` — voice files
    live ONLY inside ``voices_dir``.
    """
    if os.path.isabs(name_or_path) and os.path.isfile(name_or_path):
        return name_or_path
    base = name_or_path
    if base.endswith(".qvoice"):
        base = base[: -len(".qvoice")]
    if "/" in base or "\\" in base or ".." in base or not base:
        raise ValueError(f"invalid voice name: {name_or_path!r}")
    candidate = os.path.join(voices_dir, base + ".qvoice")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"no such voice: {candidate}")
    return candidate


def next_available_name(
    name: str,
    voices_dir: str = VOICE_DIR_DEFAULT,
    ext: str = ".qvoice",
) -> str:
    """Auto-number on collision: ``foo`` → ``foo`` / ``foo_2`` / ``foo_3``."""
    if not os.path.isfile(os.path.join(voices_dir, name + ext)):
        return name
    seq = 2
    while os.path.isfile(os.path.join(voices_dir, f"{name}_{seq}{ext}")):
        seq += 1
    return f"{name}_{seq}"
