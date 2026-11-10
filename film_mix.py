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
misplacement). A raw /v1/assets/runs or /melite-media/runs src that
skipped the walk resolves against the estate's runs trees, named by
the state_roots input FIRST (a JSON array of absolute state roots —
Layer A's submit-time walk stamps it from the estate's roots
manifest), then the standard-install derivation. The env arms are
RETIRED (2026-11-01 vendor re-cure — this pack never reads process
env; see _clip_candidate_trees).

WIRED-SEAT LAW (2026-11-01, the soundtrack merged-form wire — the
one-way law GAPS-20 face #14's film tail): a clips_json row MAY omit
src and carry ``wired: k`` (1 <= k <= WIRED_CLIP_SEATS) instead —
the card seats that clip's ``$spell`` src on this node's optional
wired_audio_k AUDIO input verbatim, the merge (rewriteSpells) lands
the producer's link in that seat, and mix_film reads the SEAT's
waveform instead of decoding a file. A path src keeps the staged-file
handoff (the lawful estate boundary, unchanged); a ``$…`` src that
reached the graph unseated refuses loud with the cure named
(_resolve_clip_source). A wired row whose seat arrived unset refuses
loud; a wired seat at a foreign sample rate refuses loud (the window
decode law — a producer riding the wire is the film's own currency,
never silently resampled).

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

# THE WIRED-CLIP SEAT COUNT (2026-11-01, the soundtrack merged-form
# wire): MeliteFilmAudioMix declares wired_audio_1..WIRED_CLIP_SEATS
# optional AUDIO inputs; a clips_json row with ``wired: k`` in
# 1..WIRED_CLIP_SEATS rides seat k instead of a staged file. The
# compose (h3-timeline actions.ts mirrors this constant) refuses a
# spelled clip beyond the seats loud — raise BOTH together.
WIRED_CLIP_SEATS = 4


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


def _clip_candidate_trees(state_roots: str = "") -> list:
    """THE §23 CROSS-ROOTS WALK (roadmap 2026-09-26 — the documented
    URL form, resolved): the estate serves run artifacts at
    ``/v1/assets/runs/<run>/<rel>`` (compose door) and
    ``/melite-media/runs/<run>/<rel>`` (browser route). A clip src in
    either spelling names a file under a state root's ``output/``
    tree. The trees, first-wins:

    - the ``state_roots`` node input's roots (J12's Layer C arm,
      2026-10-19: a JSON array of absolute state roots Layer A's
      submit-time walk stamps from the estate's roots manifest —
      call data, the same order every estate door searches, own root
      first);
    - the standard-install derivation: an engine living at
      ``<state>/runtime/comfyui`` has its runs root at
      ``<state>/runs/output`` (the estate's default layout — this
      derivation only ever ADDS a candidate; a miss refuses loud, so
      it can never smuggle a wrong file).

    VENDOR RE-CURE (2026-11-01) — COMFYUI EXTENSION NEVER USE ENV
    VARIABLES. The MELITE_STATE_ROOT / MELITE_STATE_ROOTS arms are
    retired (the estate retired its own export long ago; the operator
    seats roots by wiring ``state_roots``, never by env).
    """
    trees: list = []
    for root in _parse_state_roots(state_roots):
        trees.append(os.path.join(root, "output"))
    cwd = os.getcwd()
    if os.path.basename(cwd) == "comfyui":
        trees.append(os.path.join(os.path.dirname(os.path.dirname(cwd)), "runs", "output"))
    seen = set()
    return [t for t in trees if not (t in seen or seen.add(t))]


def _parse_state_roots(state_roots: str) -> list:
    """The state_roots node input → a list of absolute state roots.

    Empty input → []. A JSON array of strings is the stamped form;
    anything else refuses loud (never a silent fallthrough to the
    env arms — a malformed stamp is a bug to surface, not a tree to
    skip).
    """
    text = (state_roots or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except ValueError as e:
        raise RuntimeError(
            f"MeliteFilmAudioMix: state_roots is not JSON ({e}) — "
            f"Layer A's walk stamps a JSON array of absolute roots"
        )
    if not isinstance(parsed, list) or not all(
        isinstance(r, str) and r.strip() for r in parsed
    ):
        raise RuntimeError(
            "MeliteFilmAudioMix: state_roots must be a JSON array of "
            "non-empty absolute root strings"
        )
    return [r.strip() for r in parsed]


def _resolve_clip_source(src: str, state_roots: str = "") -> str:
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
            trees = _clip_candidate_trees(state_roots)
            for tree in trees:
                cand = os.path.join(tree, rel)
                if os.path.isfile(cand):
                    return cand
            searched = "; ".join(trees) if trees else (
                "no state root known — the state_roots node input is "
                "unstamped and this engine's env names no root"
            )
            raise RuntimeError(
                f"MeliteFilmAudioMix: clip source {src!r} resolves to no "
                f"file under the estate's runs trees (searched: {searched}) "
                f"— the upstream output is unreachable; re-run the producer "
                f"card, or pass the staged input name / an absolute path"
            )
    if os.path.isabs(src) and os.path.isfile(src):
        return src
    if src.startswith("$"):
        # THE OLD PATH REDS IF REACHED, WITH THE FIX NAMED (the
        # soundtrack merged-form wire, 2026-11-01): a $spell src in
        # clips_json means the soundtrack wire never seated — the card
        # embeds spelled rows as wired seats since the wire landed;
        # a graph still carrying the spell in the JSON string predates
        # it (or is hand-composed) and the merge had no seat to land
        # the producer's link in.
        raise RuntimeError(
            f"MeliteFilmAudioMix: clip source {src!r} is a $spell — the "
            f"soundtrack wire never seated (a spelled clip src names a "
            f"producer, never a file): compose the wired seat (clips_json "
            f"row wired: k + the producer linked into wired_audio_k) or "
            f"run the producer first and paste its artifact path"
        )
    raise RuntimeError(
        f"MeliteFilmAudioMix: clip source {src!r} resolves to no file "
        f"(staged input-dir name, estate asset URL, or absolute path — "
        f"never a guess)"
    )


def _trim_source_head(loaded: dict, clip: dict) -> dict:
    """THE CLIPPING LANE (2026-11-17): in_sec is the SOURCE in-point —
    where in the file the clip starts. The head trim happens BEFORE
    the delay seat, so the overlay plays [in_sec, in_sec+dur_sec] at
    the film seat: the same segment the card's conditioning slice
    played (the model hears what the film plays). A row without
    in_sec (or 0) keeps the head — byte-identical to every mix
    composed before the lane existed. A source shorter than the
    in-point yields silence for the span; the delay pads the rest.
    """
    in_sec = float(clip.get("in_sec", 0.0))
    if in_sec <= 0.0:
        return loaded
    wf = loaded["waveform"]
    sr = int(loaded["sample_rate"])
    at = int(round(in_sec * sr))
    if at >= wf.shape[-1]:
        wf = wf[..., :0]
    else:
        wf = wf[..., at:]
    return {"waveform": wf, "sample_rate": sr}


def _parse_clips(raw: str) -> list:
    """clips_json → [{src | wired, start_sec, dur_sec, in_sec?, class}] (doc order kept).

    THE WIRED-ROW LAW (2026-11-01): a row MAY omit src IFF it carries
    ``wired: k`` with 1 <= k <= WIRED_CLIP_SEATS — the card seats that
    clip's spelled src on the wired_audio_k input and the row rides the
    seat (no src key, the JSON string is dead text to the merge). Any
    other src-less row keeps the loud refusal verbatim; a row carrying
    BOTH src and wired is ambiguous and refuses loud.
    """
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
        has_src = isinstance(src, str) and bool(src.strip())
        wired = row.get("wired")
        if has_src and wired is not None:
            raise RuntimeError(
                f"MeliteFilmAudioMix: clips_json[{i}] carries BOTH src and "
                f"wired — one source per row (wired: k takes the "
                f"wired_audio_{wired!r} seat, src the staged file)"
            )
        if not has_src:
            # THE WIRED-ROW ARM: only a wired: k row in 1..WIRED_CLIP_SEATS
            # may omit src; everything else keeps the original refusal.
            if (
                not isinstance(wired, int)
                or isinstance(wired, bool)
                or not (1 <= wired <= WIRED_CLIP_SEATS)
            ):
                raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] has no src — never a silent misplacement")
        try:
            start = float(row.get("start_sec", 0.0))
            dur = float(row.get("dur_sec", 0.0))
            in_sec = float(row.get("in_sec", 0.0))
        except (TypeError, ValueError):
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] needs numeric start_sec/dur_sec")
        if not (start >= 0.0) or not (dur >= 0.0):
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] needs start_sec >= 0, dur_sec >= 0")
        if not (in_sec >= 0.0):
            raise RuntimeError(f"MeliteFilmAudioMix: clips_json[{i}] needs in_sec >= 0 (the source in-point)")
        cls = str(row.get("class", "dialogue")).strip().lower()
        if cls not in _CLIP_CLASSES:
            raise RuntimeError(
                f"MeliteFilmAudioMix: clips_json[{i}] unknown class {cls!r} — dialogue | sfx | music"
            )
        if has_src:
            clips.append({"src": src.strip(), "start_sec": start, "dur_sec": dur, "in_sec": in_sec, "class": cls})
        else:
            clips.append({"wired": wired, "start_sec": start, "dur_sec": dur, "in_sec": in_sec, "class": cls})
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
            "optional": {
                "state_roots": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": 'JSON array of absolute state roots (["/abs/root", ...]) — Layer A\'s submit-time walk stamps the estate\'s roots manifest here; the clip-source walk searches these output/ trees before any env arm',
                }),
                # THE WIRED SOUNDTRACK SEATS (2026-11-01, the soundtrack
                # merged-form wire): AUDIO is a typed socket, so a link
                # validates AUDIO→AUDIO at submit; an absent optional
                # input skips validation. The card seats a spelled clip
                # src on wired_audio_k verbatim and the merge lands the
                # producer's link there — the manifest row carries
                # `wired: k`, never src.
                **{
                    f"wired_audio_{k}": ("AUDIO", {
                        "tooltip": f"Optional wired soundtrack seat {k} — the producer's AUDIO tensor for the clips_json row carrying wired: {k} (the merge lands the upstream link here)",
                    })
                    for k in range(1, WIRED_CLIP_SEATS + 1)
                },
            },
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
                    "tooltip": 'JSON array of {src | wired, start_sec, dur_sec, in_sec?, class} — the card renders the filmMix manifest here verbatim; a wired: k row rides the wired_audio_k seat instead of a file; in_sec (optional, default 0) is the source in-point — the overlay plays [in_sec, in_sec+dur_sec], the segment the conditioning slice played',
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
        state_roots: str = "",
        wired_audio_1=None,
        wired_audio_2=None,
        wired_audio_3=None,
        wired_audio_4=None,
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

        # THE WIRED SEATS (the soundtrack merged-form wire): a
        # clips_json row with `wired: k` reads seat k's AUDIO dict —
        # the merge landed the producer's link there. A wired row
        # whose seat arrived unset refuses loud (the compose promised
        # a producer; a hand-composed graph or a seat the merge never
        # filled must not silently drop the row's voice).
        wired_seats: dict = {}
        for k, seat in ((1, wired_audio_1), (2, wired_audio_2), (3, wired_audio_3), (4, wired_audio_4)):
            if seat is not None:
                wired_seats[k] = seat
        for clip in clips:
            k = clip.get("wired")
            if k is not None and k not in wired_seats:
                raise RuntimeError(
                    f"MeliteFilmAudioMix: clips_json row wired: {k} has no "
                    f"wired_audio_{k} input — the spelled soundtrack's seat "
                    f"never landed (wire the producer's AUDIO output into "
                    f"wired_audio_{k})"
                )

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
        #    THE SEAT SOURCE (the wired-seat law): a wired row reads
        #    its seat's waveform + sample rate instead of decoding a
        #    file; a wired seat at a foreign rate refuses loud (the
        #    window decode law — the windows refuse rate drift, a
        #    wired producer rides the film's own clock, never a
        #    silent resample).
        def seat_source(clip: dict) -> dict:
            k = clip.get("wired")
            if k is None:
                waveform, sr = load_audio_file(_resolve_clip_source(clip["src"], state_roots))
                return {"waveform": waveform, "sample_rate": int(sr)}
            seat = wired_seats[k]
            sr = int(seat["sample_rate"])
            if sr != base_sr:
                raise RuntimeError(
                    f"MeliteFilmAudioMix: wired seat {k} samples at "
                    f"{sr}Hz but the film runs {base_sr}Hz — a wired "
                    f"producer rides the film's own clock (the window "
                    f"decode law; every window shares the rate)"
                )
            return {"waveform": seat["waveform"], "sample_rate": sr}

        for clip in clips:
            if clip["class"] == "music":
                continue
            loaded = seat_source(clip)
            loaded = _trim_source_head(loaded, clip)
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
            loaded = seat_source(clip)
            loaded = _trim_source_head(loaded, clip)
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
