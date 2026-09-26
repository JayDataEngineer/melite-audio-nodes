"""MeliteQVoiceCopy — the .qvoice persistence law (2026-09-25).

The operator's order: ".qvoice files are preserved like the other
media." The VoiceDesign exporter (AudiocoreVoiceStudio) mints the
voice on the ENGINE'S models mount (the documented voices root,
collision-numbered) — every OTHER artifact of the run (the sample
audio, tts flacs, films) persists in the run's own output tree where
the estate's artifact routes, run_status rows, and the cross-roots
media tab serve it. The .qvoice alone stayed outside, on the mount:
preserved on disk, invisible to the run.

This node closes that: after the studio mints the voice, ONE copy
lands under the run's output tree (dest_prefix, e.g. ``<runId>/
voices``), keeping the mount original untouched (the voices root is
the engine's reuse cache — the CLONE lanes keep reading it). The copy
is a plain file copy (no torch, no ffmpeg — never ffmpeg), loud on
every miss: a voice the run cannot show it preserved is a finding,
never a silent skip.
"""

from __future__ import annotations

import os
import shutil

# The documented engine-side voices root (the runbook's VOICES ROOT).
# Optional input overrides for nonstandard engine mounts.
DEFAULT_VOICES_ROOT = "/mnt/data/models/audio/voices"


class MeliteQVoiceCopy:
    """Copies the studio-minted .qvoice into the run's output tree."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_name": ("STRING", {
                    "tooltip": "The studio export's name (the qvoice_save name) — the mint is <name>*.qvoice, newest wins (collision-numbered mounts)",
                }),
                "dest_prefix": ("STRING", {
                    "tooltip": "Output-dir prefix for the copy, e.g. <runId>/voices",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "The studio's own audio output — ORDERING ONLY (never read): the copy provably runs after the mint",
                }),
            },
            "optional": {
                "voices_root": ("STRING", {
                    "default": "",
                    "tooltip": "Engine-side voices mount override. EMPTY = the documented root (/mnt/data/models/audio/voices) resolved at RUN time — an empty default is deliberate: a directory path defaultled into the graph is unrepresentable to the source-echo guard (attempt-018's finding: the whole design lane refused), so the fallback lives in the node, never in the graph",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "copy_qvoice"
    CATEGORY = "melite/audio"
    # THE ORPHAN LAW (attempt-019 finding, engine source read): ComfyUI
    # executes only the OUTPUT-REACHABLE subgraph — validate_prompt walks
    # output nodes' dependencies, and an unconsumed node never enters
    # the execution list. The copy IS a save node, so it takes the
    # SaveAudio/SaveVideo shape: OUTPUT_NODE roots the walk and the
    # node always runs.
    OUTPUT_NODE = True
    TITLE = "Melite QVoice Copy (voice persists in the run tree)"

    def copy_qvoice(self, voice_name, dest_prefix, audio, voices_root=DEFAULT_VOICES_ROOT):
        del audio  # ordering-only input — the studio's mint precedes us
        name = str(voice_name or "").strip()
        dest = str(dest_prefix or "").strip()
        root = str(voices_root or "").strip() or DEFAULT_VOICES_ROOT
        if not name:
            raise RuntimeError(
                "MeliteQVoiceCopy: voice_name is empty — the export's name is the search key, never a guess"
            )
        if not dest:
            raise RuntimeError(
                "MeliteQVoiceCopy: dest_prefix is empty — the copy needs its run-tree seat"
            )
        if not os.path.isdir(root):
            raise RuntimeError(
                f"MeliteQVoiceCopy: voices root {root!r} is not a directory — "
                "state the engine's voices_root override or mount it"
            )
        # Collision-numbered mints share the name prefix; newest mtime
        # is THIS run's mint (the queue serializes runs per lane — the
        # same newest-wins law as MeliteConcatVideos' counter).
        matches = []
        for entry in os.listdir(root):
            if entry.startswith(name) and entry.endswith(".qvoice"):
                full = os.path.join(root, entry)
                if os.path.isfile(full):
                    matches.append(full)
        if not matches:
            raise RuntimeError(
                f"MeliteQVoiceCopy: no .qvoice matches {name!r} under {root!r} — "
                "the studio export must mint before this copy runs (wire the "
                "studio's audio into 'audio')"
            )
        source = max(matches, key=lambda p: os.path.getmtime(p))
        # THE REUSE FLAG (commission 019's collision finding, the
        # roadmap's name-freshness admission 2026-09-25): the design
        # lane's studio ALWAYS mints a fresh export this run, and the
        # copy runs after it on the ordering wire — so a newest-match
        # whose mtime PREDATES this node's execution by minutes is a
        # stale name reused from an earlier session (the studio minted
        # nothing under this prefix, or the prefix collides across
        # sessions). Stamped, never refused: the receipt names the
        # fact; the card's teaching sends new characters to fresh
        # names. (<600s window: same-graph mint + clock tolerance.)
        import time as _time
        mtime = os.path.getmtime(source)
        reused_existing = (_time.time() - mtime) > 600.0
        import folder_paths

        out_dir = folder_paths.get_output_directory()
        target_dir = os.path.join(out_dir, *[
            part for part in dest.replace("\\", "/").split("/") if part
        ])
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, os.path.basename(source))
        shutil.copyfile(source, target)
        # THE SAVE-NODE RESULT SHAPE (attempt-020 finding): the estate
        # harvests artifacts from the history UI dict (files/audio/
        # video/... rows fetched via /view) — a bare tuple return is
        # invisible to it, so the copy sat unfetched in the engine's
        # output tree. Ride the SaveAudio convention: a "files" UI
        # row names the copy by (filename, subfolder, type) and the
        # harvest lands it in the run's own tree, listed in the run's
        # artifact rows like every other media.
        return {
            "ui": {
                "files": [{
                    "filename": os.path.basename(source),
                    "subfolder": dest,
                    "type": "output",
                }],
                # the freshness receipt: which file won, when it was
                # minted, and whether that mint predates this run.
                # THE UI LIST LAW (run-a248c5614633 receipt, 2026-09-25):
                # ComfyUI's ui merge extends LIST values — scalars raise
                # TypeError and kill the whole prompt (the live kill this
                # node's scalar receipt caused). Every entry a 1-list.
                "reused_existing": [bool(reused_existing)],
                "voice_file": [os.path.basename(source)],
                "voice_mtime": [mtime],
            },
            "result": (target,),
        }
