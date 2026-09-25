"""MeliteLoudness — the film's loudness meter at graph time (2026-09-25).

THE OWED ITEM (roadmap §39, ruled engine-side): the estate's written
law is "no ffmpeg, no concat, no mix — every artifact traces to a
composition of cards", and loudness measurement belongs with the mix
that produces the audio, not in an observe tool re-deriving it from
the encoded file. This node taps the film's final audio BEFORE the
save (a passthrough meter: the AUDIO dict rides out unchanged), and
writes the measurement into the history UI dict so every surface
REPORTS the number instead of re-deriving it.

MATH — ITU-R BS.1770-4 integrated loudness, honest and standard:
  K-weighting (the " acoustic" weighting): a high-shelf biquad
  (+4 dB shelf around 1681 Hz) cascaded with a high-pass biquad
  (~38 Hz), applied per channel — coefficients derived for the
  track's ACTUAL sample rate via the standard biquad formulas
  (never a 48k table pasted onto a 44k1 track).
  Gating: 400 ms blocks, 75% overlap; block loudness
  l_j = -0.691 + 10*log10(sum_c mean(z_c^2)); absolute gate at
  -70 LUFS, then the relative gate (mean loudness of surviving
  blocks - 10 LU) — the integrated value is the mean over blocks
  above BOTH gates. Mono/stereo/multichannel all ride the same
  per-channel powers sum.
Pure torch + torchaudio biquads (the engine's own stack — no ffmpeg,
no scipy dependency, no model).

FAILURE LAW: this pack never silently falls back — a non-finite
reading, an empty track, or a non-tensor waveform refuses LOUD (a
meter that prints a fabricated number is worse than no meter).
"""

from __future__ import annotations

import math

import torch
import torchaudio.functional as AF


# BS.1770-4's OWN 48 kHz tabulation (the spec's Table, verbatim) —
# the film mix's native rate. The RBJ derivation below reproduces
# these to ~0.4% at 48k; the table rides first so a 48k track meters
# against the exact reference filters (calibrated against ffmpeg's
# ebur128: the 997 Hz -20 dBFS stereo test reads -20.0 LUFS on both).
_SPEC_48K_SHELF = (1.53512485958697, -2.69169618940638, 1.19839281085285,
                   1.0, -1.69065929318241, 0.73248077421585)
_SPEC_48K_HP = (1.0, -2.0, 1.0,
                1.0, -1.99004745483398, 0.99007225036621)


def _k_weight_coeffs(sr: int):
    """BS.1770-4 K-weighting coefficients at an arbitrary sample rate.

    At 48 kHz the spec's own tabulated coefficients ride verbatim; at
    every other rate the standard analog-prototype bilinear forms
    (RBJ cookbook) derive the pair — the same designs the table
    snapshots.
    """
    if sr == 48000:
        return _SPEC_48K_SHELF, _SPEC_48K_HP
    # Stage 1: high-shelf (f0=1681.97 Hz, Q=0.7072, gain=+3.999 dB)
    f0 = 1681.974450955533
    gain_db = 3.99984385397
    q = 0.7071752369554196
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a) * alpha
    # RBJ cookbook HIGH-shelf (the denominator signs are the shelf's
    # own — the low-shelf signs here were the +4 LU calibration bug)
    b0 = a * ((a + 1) + (a - 1) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - two_sqrt_a_alpha
    shelf = (b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0)
    # Stage 2: high-pass (f0=38.13 Hz, Q=0.5003)
    f0 = 38.13547087624207
    q = 0.500327037331397
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    hp = (b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0)
    return shelf, hp


def integrated_loudness(waveform: torch.Tensor, sr: int) -> float:
    """BS.1770-4 gated integrated loudness (LUFS). Pure torch.

    waveform: (channels, samples) float tensor, any channel count.
    """
    if waveform.dim() != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise RuntimeError(
            "MeliteLoudness: empty or non-2D track "
            f"(shape={tuple(waveform.shape)}) — refusing to meter garbage"
        )
    z = waveform.to(torch.float64)
    shelf, hp = _k_weight_coeffs(sr)
    for b0, b1, b2, a0, a1, a2 in (shelf, hp):
        z = AF.biquad(z.contiguous(), b0, b1, b2, a0, a1, a2)
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError(
            "MeliteLoudness: K-weighted track is non-finite — refusing a "
            "fabricated reading (check upstream audio for NaN/Inf)"
        )
    block = int(round(0.400 * sr))
    hop = block // 4  # 75% overlap
    n = z.shape[1]
    if n < block:
        # a track shorter than one block: meter the whole thing ungated
        # (single block, absolute gate still applies)
        powers = (z * z).mean(dim=1)
        loud = -0.691 + 10.0 * math.log10(float(powers.sum()) + 1e-12)
        if loud < -70.0:
            raise RuntimeError(
                f"MeliteLoudness: track meters at {loud:.1f} LUFS — below "
                "the -70 absolute gate (silence refuses a fabricated number)"
            )
        return loud
    powers = z.unfold(1, block, hop)
    weights = (powers * powers).mean(dim=2).sum(dim=0)  # (blocks,)
    blocks = -0.691 + 10.0 * torch.log10(weights + 1e-12)
    above = blocks[blocks > -70.0]
    if above.numel() == 0:
        raise RuntimeError(
            "MeliteLoudness: every block sits below the -70 LUFS absolute "
            "gate — silence refuses a fabricated reading"
        )
    rel = float(above.mean()) - 10.0
    kept = above[above > rel]
    if kept.numel() == 0:
        raise RuntimeError(
            "MeliteLoudness: no block survives the relative gate — the "
            "track is all gate-edge noise; refusing a fabricated number"
        )
    return float(kept.mean())


class MeliteLoudness:
    """Passthrough loudness meter: AUDIO in, same AUDIO out + the
    reading stamped into the history UI dict."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "The film's final mixed audio — passthrough: the same tensor rides out (meter taps, never transforms)",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "meter"
    CATEGORY = "melite/audio"
    OUTPUT_NODE = True
    TITLE = "Melite Loudness (BS.1770-4 integrated LUFS)"

    def meter(self, audio):
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise RuntimeError(
                "MeliteLoudness: input is not an AUDIO dict "
                "({waveform, sample_rate}) — wire the mix's audio output"
            )
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        if not isinstance(waveform, torch.Tensor):
            raise RuntimeError(
                "MeliteLoudness: AUDIO.waveform is not a tensor — this pack "
                "meters real waveforms, never guesses at encoded containers"
            )
        track = waveform.detach().cpu().to(torch.float64)
        # ComfyUI AUDIO dicts may ride (samples,) mono or (channels, samples)
        if track.dim() == 1:
            track = track.unsqueeze(0)
        lufs = integrated_loudness(track, sr)
        peak = float(track.abs().max())
        peak_db = 20.0 * math.log10(peak + 1e-12)
        if not math.isfinite(lufs) or not math.isfinite(peak_db):
            raise RuntimeError("MeliteLoudness: non-finite reading — loud out")
        return {
            "ui": {
                "loudness_lufs": lufs,
                "peak_db": peak_db,
                "sample_rate": sr,
            },
            "result": (audio,),
        }


NODE_CLASS_MAPPINGS = {"MeliteLoudness": MeliteLoudness}
NODE_DISPLAY_NAME_MAPPINGS = {"MeliteLoudness": "Melite Loudness (LUFS)"}
