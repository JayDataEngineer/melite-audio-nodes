"""Family engines — the modules core.py dispatches into by direct import.

The registry that once lived here (get_engine_class/python_families,
always returning nothing) is deleted: no caller existed, and its
docstring claimed all inference rode the audiocpp_server subprocess
— false since the moss_sfx_v2 cure (core.py imports
engines.moss_sfx_v2 directly and loads a pure in-process torch
pipeline; only the qwen3_tts voice-design path speaks HTTP to the
server). The modules:

- engines/qwen3_tts.py — the qwen-tts Python package seat: the
  VoiceStudio node imports it for voice export/preview, and the
  voice-design pipeline drives it.
- engines/moss_sfx_v2.py — the MOSS-SFX v2 torch engine
  (core.py's dispatch target for family moss_sfx_v2).
- engines/moss_sfx_impl/ — the vendored upstream implementation
  tree moss_sfx_v2 loads through.
"""
