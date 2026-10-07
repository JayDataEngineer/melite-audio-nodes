"""install.py — the ComfyUI-Manager install door for the native audio core.

A pack install must converge the pack: Manager clones this repo and
runs this file, which fetches the pinned libaudiocore_native.so
release asset into this pack's own native/ directory (sha256-verified
— a drifted asset is a refusal, never a silent pass). No estate
tooling, no environment variables, no manual steps: after Manager
installs this pack and the models download, the nodes load.

stdlib only — this runs before pip requirements converge.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

TAG = "libaudiocore-3938031"
URL = (
    "https://github.com/JayDataEngineer/audio.cpp/releases/download/"
    + TAG
    + "/libaudiocore_native.so"
)
SHA256 = "4a7069054ba608ce3bd099e09106d6036984023e657fdf60de8ff3394e18a590"

NATIVE_DIR = Path(__file__).resolve().parent / "native"
NATIVE_LIB = NATIVE_DIR / "libaudiocore_native.so"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if NATIVE_LIB.is_file() and _sha256(NATIVE_LIB) == SHA256:
        print(f"[melite audio-core] native core already converged: {NATIVE_LIB}")
        return 0
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = NATIVE_LIB.with_suffix(".download")
    print(f"[melite audio-core] fetching native core {TAG} …")
    try:
        urllib.request.urlretrieve(URL, tmp)
    except Exception as exc:  # noqa: BLE001 — report every network shape once
        print(
            f"[melite audio-core] FAILED to fetch {URL}: {exc}\n"
            "The native core is a pinned release asset of "
            "github.com/JayDataEngineer/audio.cpp — retry the install, or "
            "set AUDIOCORE_NATIVE_LIB to a .so you built.",
            file=sys.stderr,
        )
        return 1
    got = _sha256(tmp)
    if got != SHA256:
        tmp.unlink(missing_ok=True)
        print(
            f"[melite audio-core] REFUSED: sha256 mismatch for {TAG}\n"
            f"  expected {SHA256}\n  got      {got}\n"
            "A drifted asset never installs. Re-pin the pack.",
            file=sys.stderr,
        )
        return 1
    tmp.replace(NATIVE_LIB)
    print(f"[melite audio-core] native core installed: {NATIVE_LIB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
