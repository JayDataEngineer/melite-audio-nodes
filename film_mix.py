"""MeliteFilmAudioMix — the valve's second arm (2026-09-24): the film's
post-generation audio mix as a vendored ComfyUI node (Layer C).

THE LAW (operator, 2026-09-24): WE USE COMFYUI, NEVER FFMPEG for the
mix. Films WITH positioned audio_clips kept the in-graph film stitch
(the film-level mix + duck bus), so every window stayed resident until
the final CreateVideo — the VRAM accumulation that OOM'd a two-window
BRIEF9 film holding 20.57 GiB (27 clips-free precedent films never
OOM'd: the disk-stitch valve frees each window after its SaveVideo).
The fix: audio-carrying films take the valve path too (windows save +
free per-window behind MeliteUnload boundaries), and THIS node composes
the film's audio post-generation from the saved window FILES + the
pinned clips — pure torch waveform math, no model, no ffmpeg
filtergraph anywhere. Output feeds SaveAudio (the film's final mux).

SEMANTICS — ported exactly from the retired in-graph tail
(packages/video/h3-timeline/src/card.ts + src/filmMix.ts): the node
invokes the SAME MeliteAudioDelay / MeliteAudioMix / MeliteAudioDuck
methods the graph used to call, in the SAME order (dialogue/sfx
placements onto the film track in doc order, then the music sub-bus
built in doc order, ducked, joined last):

  generated track = the window audio files decoded and BUTT-JOINED in
      film order (torch.cat on time — the old MeliteConcatVideos audio
      axis), then fit to film_seconds (trim / silence-pad — output
      length = film length, encoder padding never rides);
  the track DUCKS (duck_db; 20ms attack, 350ms release — the console
      law, card-hardcoded before, node-owned now) during every
      conditioned span (duck_spans, "start:end;…" seconds, ms
      precision);
  music clips pre-mix into their own sub-bus (length_mode 'base'),
      duck under the dialogue+sfx spans ONLY (music_duck_spans), then
      join the film at unity;
  dialogue/sfx clips overlay pristine at their seats: head silence to
      start_sec (sample-grain at the clip's own rate) + exact
      seat+dur fit (ms-rounded, the card's own law); dur 0 = unfitted
      (full length, never a fit);
  every placement mixes with length_mode 'base' — a clip never
      extends the cut; the sum clamps to [-1,1] (every AUDIO
      producer's convention in this pack).

WINDOW-FILE LAW (the MeliteConcatVideos contract, mirrored): `videos`
is a comma list of output-dir SaveVideo prefixes in film order; each
resolves at runtime (post-save) to the newest `<prefix>_<counter>_.mp4`
and refuses LOUD on no match (a missing window is never skipped).
Mismatched sample rates across windows refuse loud (never a silent
resample); channels broadcast to the widest window (mono repeats).
`after` is the MeliteUnload boundary token (REQUIRED) — the ordering
proof every save completed; empty refuses loud (never a stale stitch).

CLIP-SOURCE LAW: each clips_json row carries the staged filename the
card composed (the submit-time upload walk stages data:/asset URLs to
input-dir names) or an absolute path; resolution is
get_annotated_filepath first, absolute-path fallback second, loud
refusal otherwise. Bad JSON / bad rows refuse loud (never a silent
misplacement).

TWO HONEST DIVERGENCES from the retired tail (both improvements, both
loud by construction):
  - a window file with NO audio track refuses (the old concat
    substituted silent stereo — silence for a missing performance is
    the exact silent-wrong-film class this pack bans);
  - the generated track fits film_seconds, so AAC priming samples the
    old butt-join carried no longer stretch the film's audio.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import re

import torch

try:
    import folder_paths
except ImportError:  # standalone tooling/tests, never inside ComfyUI
    folder_paths = None

_COUNTER_RE = re.compile(r"_(\d{5})_\.mp4$")

# the console law (operator catch c, 2026-09-22): a duck attacks fast
# (20ms) and RELEASES slow (350ms — a both-edges 20ms duck popped the
# bed back like a gate). Card-hardcoded in the retired tail; node-owned
# now — one seat for the edge shape, the depth stays a knob (duck_db).
_ATTACK_MS = 20.0
_RELEASE_MS = 350.0

_CLIP_CLASSES = ("dialogue", "sfx", "music")


def _resolve_output_prefix(prefix: str) -> str:
    """One output-dir prefix → its newest ``<prefix>_<counter>_.mp4``.

    The MeliteConcatVideos lookup law, mirrored (that pack owns the
    VIDEO tail; this node owns the AUDIO tail — same files, same
    newest-by-counter rule): SaveVideo writes
    ``<prefix>_<counter:05>_.mp4`` with a process-monotonic counter, so
    newest-by-counter is THIS run's write. Loud refusal on no match.
    """
    if folder_paths is None:
        raise RuntimeError("MeliteFilmAudioMix requires ComfyUI (folder_paths)")
    out_dir = folder_paths.get_output_directory()
    pattern = os.path.join(out_dir, f"{prefix}_*.mp4")
    matches = [p for p in _glob.glob(pattern) if _COUNTER_RE.search(p) is not None]
    if not matches:
        raise RuntimeError(
            f"MeliteFilmAudioMix: no saved video matches prefix "
            f"{prefix!r} under the output directory ({pattern}) — the "
            f"window save must complete before this mix runs (wire "
            f"MeliteUnload's token into 'after')"
        )

    def counter_of(path: str) -> int:
        return int(_COUNTER_RE.search(path).group(1))  # type: ignore[union-attr]

    return max(matches, key=counter_of)


def _clip_candidate_trees() -> list:
    """THE §23 CROSS-ROOTS WALK (roadmap 2026-09-26 — the documented
    URL form, resolved): the estate serves run artifacts at
    ``/v1/assets/runs/<run>/<rel>`` (compose door) and
    ``/melite-media/runs/<run>/<rel>`` (browser route). A clip src in
    either spelling names a file under a state root's ``output/``
    tree. The trees, first-wins (the same order every estate door
    searches — own root first, colon-separated MELITE_STATE_ROOTS
    extras after):

    - ``$MELITE_STATE_ROOT/output`` when the env names the root;
    - every ``$MELITE_STATE_ROOTS`` extra's ``output/``;
    - the standard-install derivation: an engine living at
      ``<state>/runtime/comfyui`` has its runs root at
      ``<state>/runs/output`` (the estate's default layout — this
      derivation only ever ADDS a candidate; a miss refuses loud, so
      it can never smuggle a wrong file).
    """
    trees: list = []
    env_root = os.environ.get("MELITE_STATE_ROOT", "").strip()
    if env_root:
        trees.append(os.path.join(env_root, "output"))
    for extra in os.environ.get("MELITE_STATE_ROOTS", "").split(":"):
        extra = extra.strip()
        if extra:
            trees.append(os.path.join(extra, "output"))
    cwd = os.getcwd()
    if os.path.basename(cwd) == "comfyui":
        trees.append(os.path.join(os.path.dirname(os.path.dirname(cwd)), "runs", "output"))
    seen = set()
    return [t for t in trees if not (t in seen or seen.add(t))]


def _resolve_clip_source(src: str) -> str:
    """A clips_json src → a decodable path (staged name, then the
    estate's asset-URL form, then absolute)."""
    if folder_paths is None:
        raise RuntimeError("MeliteFilmAudioMix requires ComfyUI (folder_paths)")
    try:
        staged = folder_paths.get_annotated_filepath(src)
    except Exception:
        staged = None
    if staged is not None and os.path.isfile(staged):
        return staged
    # THE ASSET-URL ARM (2026-09-26): the h3 tool description
    # advertises ``/v1/assets/runs/<run>/<rel>`` FIRST — resolve it
    # against the state trees instead of dying with a form the docs
    # told the author to use. Layer A's submit-time walk already
    # stages these URLs into input-dir names; this arm is the defense
    # for any src that reached the graph raw (a hand-composed graph,
    # or a future seat that skips the walk).
    for prefix in ("/v1/assets/runs/", "/melite-media/runs/"):
        if src.startswith(prefix):
            rel = src[len(prefix):]
            if ".." in rel.split("/"):
                raise RuntimeError(
                    f"MeliteFilmAudioMix: clip source {src!r} refuses traversal"
                )
            trees = _clip_candidate_trees()
            for tree in trees:
                cand = os.path.join(tree, rel)
                if os.path.isfile(cand):
                    return cand
            searched = "; ".join(trees) if trees else "no state root known (set MELITE_STATE_ROOT)"
            raise RuntimeError(
                f"MeliteFilmAudioMix: clip source {src!r} resolves to no "
                f"file under the estate's runs trees (searched: {searched}) "
                f"— the upstream output is unreachable; re-run the producer "
                f"card, or pass the staged input name / an absolute path"
            )
    if os.path.isabs(src) and os.path.isfile(src):
        return src
    raise RuntimeError(
        f"MeliteFilmAudioMix: clip source {src!r} resolves to no file "
        f"(staged input-dir name, estate asset URL, or absolute path — "
        f"never a guess)"
    )


def _parse_clips(raw: str) -> list:
    """clips_json → [{src, start_sec, dur_sec, class}] (doc order kept)."""
    try:
        rows = json.loads(raw or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"MeliteFilmAudioMix: clips_json is not JSON ({e})")
    if not isinstance(rows, list):
        raise RuntimeError("MeliteFilmAudioMix: clips_json must be a JSON array of clip rows")
    clips = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] is not an object — never a silent skip")
        src = row.get("src")
        # THE FIRST-LIVE-RUN TYPO (attempt-017, paid for live): this seat
        # shipped `isinstance(src)` — one argument — and killed the film at
        # node 47 with "isinstance expected 2 arguments, got 1". The type
        # argument was always the law.
        if not isinstance(src, str) or not src.strip():
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] has no src — never a silent misplacement")
        try:
            start = float(row.get("start_sec", 0.0))
            dur = float(row.get("dur_sec", 0.0))
        except (TypeError, ValueError):
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] needs numeric start_sec/dur_sec")
        if not (start >= 0.0) or not (dur >= 0.0):
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] needs start_sec >= 0, dur_sec >= 0")
        cls = str(row.get("class", "dialogue")).strip().lower()
        if cls not in _CLIP_CLASSES:
            raise RuntimeError(
                f"MeliteFilmAudioMix: clips_json[{i}] unknown class {cls!r} — dialogue | sfx | music"
            )
        clips.append({"src": src.strip(), "start_sec": start, "dur_sec": dur, "class": cls})
    return clips


class MeliteFilmAudioMix:
    """Compose the film's audio post-generation: window files + pinned clips + duck spans."""

    CATEGORY = "audio/melite"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "mix_film"
    DESCRIPTION = (
        "Post-generation film audio mix (the valve's second arm): butt-join the saved window "
        "audios, duck the generated track under conditioned spans, overlay positioned "
        "dialogue/sfx pristine, pre-mix + duck the music sub-bus. Torch only, never ffmpeg. "
        "Output is SaveAudio-ready; output length = film_seconds."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma list of output-dir SaveVideo prefixes, in film order (e.g. h3_t4_w1,h3_t4_w2)",
                }),
                "after": ("STRING", {
                    "tooltip": "The MeliteUnload boundary token — proof every window save completed",
                }),
                "clips_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": 'JSON array of {src, start_sec, dur_sec, class} — the card renders the filmMix manifest here verbatim',
                }),
                "duck_spans": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Semicolon list of start:end second spans ducking the generated track (conditioned clips)",
                }),
                "music_duck_spans": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Semicolon list of start:end second spans ducking the music sub-bus (dialogue/sfx only)",
                }),
                "duck_db": ("FLOAT", {
                    "default": -14.0, "min": -60.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Duck depth in dB (the card's duck_depth_db knob, verbatim)",
                }),
                "film_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "step": 0.001,
                    "tooltip": "Total film length in seconds (the manifest's filmSec) — the output fits exactly",
                }),
            },
        }

    def mix_film(
        self,
        videos: str,
        after: str,
        clips_json: str,
        duck_spans: str,
        music_duck_spans: str,
        duck_db: float,
        film_seconds: float,
    ):
        # the engine half lives in .nodes (Delay/Mix/Duck + the PyAV
        # decode law) — imported HERE, not at module top: nodes.py
        # imports this module at ITS top for registration, and this
        # module must never import .nodes at ITS top (the cycle).
        from .nodes import MeliteAudioDelay, MeliteAudioDuck, MeliteAudioMix, load_audio_file

        prefixes = [p.strip() for p in str(videos).split(",") if p.strip()]
        if len(prefixes) == 0:
            raise RuntimeError(
                "MeliteFilmAudioMix: 'videos' is empty — the film mix needs "
                "at least one saved window prefix"
            )
        if not str(after).strip():
            raise RuntimeError(
                "MeliteFilmAudioMix: 'after' is empty — wire MeliteUnload's "
                "token (the ordering proof that every save completed)"
            )
        if not (film_seconds > 0.0):
            raise RuntimeError(
                "MeliteFilmAudioMix: film_seconds must be positive — the output fits the film exactly"
            )
        clips = _parse_clips(clips_json)

        # 1. THE GENERATED TRACK: decode every window file, refuse rate
        #    drift (never a silent resample — the concat's own law),
        #    broadcast channels to the widest window, butt-join on time.
        tracks = []
        base_sr = None
        for prefix in prefixes:
            waveform, sr = load_audio_file(_resolve_output_prefix(prefix))
            if base_sr is None:
                base_sr = int(sr)
            elif int(sr) != base_sr:
                raise RuntimeError(
                    f"MeliteFilmAudioMix: window {prefix!r} samples at "
                    f"{int(sr)}Hz but the film runs {base_sr}Hz — every "
                    f"window must share the rate"
                )
            tracks.append(waveform)
        assert base_sr is not None
        max_ch = max(int(w.shape[1]) for w in tracks)
        aligned = []
        for w in tracks:
            if int(w.shape[1]) < max_ch:
                w = w.repeat_interleave(max_ch // int(w.shape[1]), dim=1)
            aligned.append(w)
        film = torch.cat(aligned, dim=2)
        # fit the film length exactly (trim / silence-pad — the Delay
        # exact-fit law; encoder padding never rides the film).
        expected = int(round(float(film_seconds) * base_sr))
        if int(film.shape[-1]) > expected:
            film = film[..., :expected]
        elif int(film.shape[-1]) < expected:
            film = torch.nn.functional.pad(film, (0, expected - int(film.shape[-1])))
        film_audio = {"waveform": film, "sample_rate": base_sr}

        duck = MeliteAudioDuck()
        delay = MeliteAudioDelay()
        mix = MeliteAudioMix()

        # 2. THE DUCK MATRIX (D-M-E, the manifest's voice): the
        #    generated track yields under every conditioned span.
        if (duck_spans or "").strip():
            film_audio = duck.duck_audio(
                film_audio,
                segments=duck_spans,
                gain_db=float(duck_db),
                ramp_ms=_ATTACK_MS,
                release_ms=_RELEASE_MS,
            )[0]

        # 3. THE PLACEMENTS (the retired tail's two loops, verbatim):
        #    dialogue/sfx onto the film track in doc order, then the
        #    music sub-bus (doc order, 'base'), ducked under voice
        #    spans only, joined last. duration fit = round(seat+dur to
        #    ms); dur 0 = unfitted (the card's own law, both arms).
        for clip in clips:
            if clip["class"] == "music":
                continue
            waveform, sr = load_audio_file(_resolve_clip_source(clip["src"]))
            loaded = {"waveform": waveform, "sample_rate": int(sr)}
            placed = delay.delay_audio(
                loaded,
                delay_seconds=float(clip["start_sec"]),
                duration_seconds=round((float(clip["start_sec"]) + float(clip["dur_sec"])) * 1000) / 1000
                if float(clip["dur_sec"]) > 0
                else 0,
            )[0]
            film_audio = mix.mix_audio(film_audio, placed, length_mode="base")[0]
        music_bus = None
        for clip in clips:
            if clip["class"] != "music":
                continue
            waveform, sr = load_audio_file(_resolve_clip_source(clip["src"]))
            loaded = {"waveform": waveform, "sample_rate": int(sr)}
            placed = delay.delay_audio(
                loaded,
                delay_seconds=float(clip["start_sec"]),
                duration_seconds=round((float(clip["start_sec"]) + float(clip["dur_sec"])) * 1000) / 1000
                if float(clip["dur_sec"]) > 0
                else 0,
            )[0]
            music_bus = placed if music_bus is None else mix.mix_audio(music_bus, placed, length_mode="base")[0]
        if music_bus is not None:
            if (music_duck_spans or "").strip():
                music_bus = duck.duck_audio(
                    music_bus,
                    segments=music_duck_spans,
                    gain_db=float(duck_db),
                    ramp_ms=_ATTACK_MS,
                    release_ms=_RELEASE_MS,
                )[0]
            film_audio = mix.mix_audio(film_audio, music_bus, length_mode="base")[0]
        return (film_audio,)
