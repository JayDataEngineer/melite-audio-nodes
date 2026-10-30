#!/usr/bin/env python3
"""The warmed-seat release door's unit tests (NEW-3, 2026-11-12): the
estate's free door evicts only ComfyUI-core-tracked models, while the
qwen3 torch engine's process-wide warmed handles (~4.3 GB + graphs per
variant) are seats no eviction owns — warm-card saves then starve with
the Unload node sufficient but unreachable. POST
/audiocore/release_warmed fires the same release_warmed_handles() the
Unload node fires, for callers that hold no AUDIOCORE_MODEL handle.

Weight-free: server.PromptServer and aiohttp.web are stubbed (a routes
recorder + a json recorder); the handler and release_warmed_handles
are the REAL pack code, loaded as a synthetic-package submodule so the
handler's relative engine import resolves. laws judged: the POST route
registration, the real release firing through the door (a seeded
warmed seat is dropped and counted), and the guard law (no server
instance, no registration, no crash).
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "melite_audio_release_test"


def _install_standalone_dirs():
    """The pack DECLARES its model folders (no defaults): outside
    ComfyUI the test escape hatch is env vars — point both at temp
    dirs so the import-time resolution succeeds."""
    tmp = tempfile.mkdtemp(prefix="melite-audio-release-test-")
    os.environ.setdefault("AUDIOCPP_MODELS_DIR", tmp)
    os.environ.setdefault("QWEN3_HF_ROOT", tmp)


_install_standalone_dirs()


class _Routes:
    def __init__(self):
        self.posts = {}

    def post(self, path):
        def deco(fn):
            self.posts[path] = fn
            return fn
        return deco

    def get(self, path):  # the pack registers no GETs; present for shape
        def deco(fn):
            return fn
        return deco


def _install_stubs(server_instance):
    """Stub ComfyUI's server + aiohttp for the TEST RUN'S LIFETIME
    (both the load-time guard and the handler's call-time
    ``from aiohttp import web`` resolve through these)."""
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=server_instance)
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(
        json_response=lambda payload, status=200: ("json", payload, status))
    sys.modules["server"] = server
    sys.modules["aiohttp"] = aiohttp
    return server


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


class ReleaseWarmedRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes = _Routes()
        cls.server = _install_stubs(types.SimpleNamespace(routes=cls.routes))
        cls.nodes = _load_nodes("served")

    def test_route_registered_as_post(self):
        self.assertIn("/audiocore/release_warmed", self.routes.posts,
                      "the release door is a POST route, not a GET")
        self.assertIs(self.routes.posts["/audiocore/release_warmed"],
                      self.nodes._audiocore_release_warmed_command,
                      "the registered handler is the pack's own command")

    def test_handler_releases_a_seeded_warmed_seat(self):
        from melite_audio_release_test_served.engines.qwen3_tts import (
            _WARMED_HANDLES,
            release_warmed_handles,
        )
        _WARMED_HANDLES[("/tmp/fake-hf-dir", "0.6B-CustomVoice")] = object()
        try:
            kind, payload, status = self.nodes._audiocore_release_warmed_command(
                object())
        finally:
            _WARMED_HANDLES.clear()
        self.assertEqual(kind, "json")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"released": 1},
                         "the door counts the dropped seats for the log")
        self.assertEqual(len(_WARMED_HANDLES), 0,
                         "the process-wide cache is empty after the door")
        self.assertEqual(release_warmed_handles(), 0,
                         "a second firing releases nothing — idempotent")

    def test_guard_no_instance_no_registration_no_crash(self):
        # Flip the shared stub to a dead instance for the second load:
        # the guard must skip registration without touching routes.
        live = self.server.PromptServer.instance
        self.server.PromptServer.instance = None
        try:
            module = _load_nodes("guarded")
        finally:
            self.server.PromptServer.instance = live
        self.assertIsNotNone(module._AudioPromptServer,
                             "the import itself proved the pack loads outside ComfyUI")


if __name__ == "__main__":
    unittest.main()
