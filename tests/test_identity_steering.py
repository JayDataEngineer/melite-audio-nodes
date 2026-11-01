#!/usr/bin/env python3
"""The TTS mod-surface cures' unit walls (2026-11-01, the audit quartet).

F51 — voice_strength is an IDENTITY SCALE: the preview path steers each
speaker embedding along its deviation from the voices bank's neutral
(``steered = neutral + (emb - neutral) * strength`` — the PCA steering
law), never the raw ``emb * strength`` rescale that re-rolled the whole
performance (locked-seed corr 0.10). Walls: the steering math, the
s=1.0 identity short-circuit (the voice LOCK), the bank mean build
(same-dim lite siblings; foreign dims and unreadable files skipped),
the cache refresh, and the loud no-bank degrade to identity.

F50 — the AudiocoreTTS required block carries only what the node
demands: no phantom ``speaker_name`` requirement (the tts card's seat
is ``voice``; compose never sends speaker_name — "" rides and the
engine's voice-source gate refuses loud when a take has no voice at
all).

Weight-free: the same synthetic-package + stub harness as
_audiocore_release_unit_test.py; no model loads, no GPU.
"""

import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "melite_audio_steering_test"


def _install_folder_paths():
    tmp = tempfile.mkdtemp(prefix="melite-audio-steering-test-")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {
        "audiocpp": ([tmp], {".gguf"}),
        "qwen3_tts": ([tmp], set()),
    }
    folder_paths.get_folder_paths = lambda key: [tmp]
    sys.modules["folder_paths"] = folder_paths


_install_folder_paths()


def _install_stubs():
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=None)
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace()
    sys.modules["server"] = server
    sys.modules["aiohttp"] = aiohttp


_install_stubs()


def _load_engine(tag):
    name = f"{PKG}_{tag}"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{name}.engines.qwen3_tts", ROOT / "engines" / "qwen3_tts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}.engines.qwen3_tts"] = module
    spec.loader.exec_module(module)
    return module


class _Item:
    """A stand-in VoiceClonePromptItem (write_lite reads attributes)."""

    def __init__(self, embedding, ref_code=None):
        self.ref_code = ref_code
        self.ref_spk_embedding = embedding
        self.x_vector_only_mode = False
        self.icl_mode = False
        self.ref_text = None


def _write_voice(path, embedding, variant="base-0.6b"):
    import torch
    qvoice_spec = importlib.util.spec_from_file_location(
        f"{PKG}_engine.qvoice", ROOT / "qvoice.py")
    qvoice = importlib.util.module_from_spec(qvoice_spec)
    sys.modules[f"{PKG}_engine.qvoice"] = qvoice
    qvoice_spec.loader.exec_module(qvoice)
    qvoice.write_lite(
        path, name=Path(path).stem, instruct="", sample_text="probe",
        items=[_Item(torch.tensor(embedding, dtype=torch.float32))],
        export_variant=variant,
    )


class IdentitySteeringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = _load_engine("engine")
        cls.engine = object.__new__(cls.eng.Qwen3TtsEngine)

    def test_steer_scales_the_deviation_from_neutral(self):
        import torch
        emb = torch.tensor([2.0, 0.0], dtype=torch.float32)
        neutral = torch.tensor([0.0, 0.0], dtype=torch.float32)
        bank = tempfile.mkdtemp(prefix="steer-math-")
        path = str(Path(bank) / "v.qvoice")
        _write_voice(path, [0.0, 0.0])
        eng = self.engine
        # Neutral rides the cache seam through a stub — the math is the wall.
        eng._bank_neutral_embedding = lambda e, p: neutral  # noqa: B010
        s14 = eng._steer_identity_strength(emb.clone(), 1.4, path)
        s05 = eng._steer_identity_strength(emb.clone(), 0.5, path)
        expected14 = neutral + (emb - neutral) * 1.4
        expected05 = neutral + (emb - neutral) * 0.5
        self.assertTrue(torch.allclose(s14, expected14, atol=1e-6),
                        "1.4 exaggerates: the deviation from neutral scales by 1.4")
        self.assertTrue(torch.allclose(s05, expected05, atol=1e-6),
                        "0.5 pulls toward neutral: the deviation scales by 0.5")

    def test_strength_one_returns_the_input_object(self):
        import torch
        emb = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        bank = tempfile.mkdtemp(prefix="steer-lock-")
        path = str(Path(bank) / "v.qvoice")
        _write_voice(path, [0.0, 0.0, 0.0])
        out = self.engine._steer_identity_strength(emb, 1.0, path)
        self.assertIs(out, emb,
                      "strength 1.0 is the LOCK: the saved embedding rides untouched")

    def test_no_bank_rides_identity(self):
        import torch
        emb = torch.tensor([1.0, 2.0], dtype=torch.float32)
        empty = tempfile.mkdtemp(prefix="steer-empty-")
        path = str(Path(empty) / "only.qvoice")
        _write_voice(path, [1.0, 2.0])
        # The bank mean excludes nothing here — but a foreign-dim-only bank
        # yields None: point the probe at a 3-dim embedding against this
        # 2-dim bank.
        emb3 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        out = self.engine._steer_identity_strength(emb3, 1.4, path)
        self.assertIs(out, emb3,
                      "no same-dim bank mean: identity rides (loud warning), never a re-roll")


class BankNeutralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = _load_engine("bank")
        cls.engine = object.__new__(cls.eng.Qwen3TtsEngine)

    def test_bank_mean_from_same_dim_lite_siblings(self):
        import torch
        bank = Path(tempfile.mkdtemp(prefix="bank-mean-"))
        _write_voice(str(bank / "a.qvoice"), [2.0, 4.0])
        _write_voice(str(bank / "b.qvoice"), [4.0, 6.0])
        _write_voice(str(bank / "foreign.qvoice"), [1.0, 2.0, 3.0],
                     variant="base-1.7b")
        (bank / "corrupt.qvoice").write_bytes(b"QVOICE-LITE\x00\x00\x00\x00\x00junk")
        emb = torch.tensor([0.0, 0.0], dtype=torch.float32)
        mean = self.engine._bank_neutral_embedding(emb, str(bank / "a.qvoice"))
        self.assertIsNotNone(mean, "two same-dim siblings: a mean exists")
        expected = torch.tensor([3.0, 5.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(mean, expected, atol=1e-6),
                        "the mean averages the same-dim members only — "
                        "the foreign dim and the corrupt file are not bank members")

    def test_cache_refreshes_when_the_bank_moves(self):
        import torch
        bank = Path(tempfile.mkdtemp(prefix="bank-cache-"))
        _write_voice(str(bank / "a.qvoice"), [2.0, 2.0])
        emb = torch.tensor([0.0, 0.0], dtype=torch.float32)
        path = str(bank / "a.qvoice")
        first = self.engine._bank_neutral_embedding(emb, path)
        self.assertTrue(torch.allclose(first, torch.tensor([2.0, 2.0]), atol=1e-6))
        _write_voice(str(bank / "b.qvoice"), [4.0, 4.0])
        second = self.engine._bank_neutral_embedding(emb, path)
        self.assertTrue(torch.allclose(second, torch.tensor([3.0, 3.0]), atol=1e-6),
                        "a new bank member moves the newest-mtime stamp — the "
                        "cached mean refreshes, never a stale neutral")

    def test_empty_bank_is_no_mean(self):
        import torch
        bank = Path(tempfile.mkdtemp(prefix="bank-empty-"))
        emb = torch.tensor([1.0], dtype=torch.float32)
        self.assertIsNone(self.engine._bank_neutral_embedding(emb, str(bank / "x.qvoice")),
                          "no bank, no anchor — the caller rides identity")


class SpeakerRequiredBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = _load_nodes("nodes")

    def test_required_block_carries_no_speaker(self):
        inp = self.nodes.AudiocoreTTS.INPUT_TYPES()
        self.assertNotIn("speaker_name", inp["required"],
                         "F50: the node never demands a speaker — a required "
                         "speaker_name is the phantom the card's form rendered")
        self.assertIn("speaker_name", inp["optional"],
                      "the speaker seat stays available as an optional input")
        self.assertEqual(inp["optional"]["speaker_name"][1]["default"], "",
                         "the optional seat keeps its empty default (compose "
                         "never sends it; the engine gate refuses loud when a "
                         "take has no voice source at all)")

    def test_studio_voice_strength_documents_the_identity_scale(self):
        inp = self.nodes.AudiocoreVoiceStudio.INPUT_TYPES()
        tooltip = inp["optional"]["voice_strength"][1].get("tooltip", "")
        self.assertIn("identity", tooltip,
                      "the node face states the F51 truth: an identity scale")
        self.assertIn("neutral", tooltip,
                      "the tooltip names the neutral anchor the steering rides")


def _load_nodes(tag):
    name = f"{PKG}_{tag}"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{name}.nodes", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}.nodes"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
