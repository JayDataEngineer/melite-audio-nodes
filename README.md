# melite-audio-nodes

audio lane nodes (VAE audio decode, audio I/O helpers).

audiocpp-fork — ComfyUI custom nodes for the audiocpp-fork C++ audio engine.

## Nodes

- `AudiocoreFamilyInfo`
- `AudiocoreMusic`
- `AudiocoreTTS`
- `AudiocoreVoiceEmbedding`
- `AudiocoreVoiceStudio`
- `LoadAudiocoreModel`
- `UnloadAudiocoreModel`

## Install

Install through ComfyUI-Manager (git URL
`https://github.com/JayDataEngineer/melite-audio-nodes`). Manager
runs `install.py`, which fetches + sha-verifies the pinned native
library release asset (`libaudiocore-3938031`) into this pack's
`native/` dir — no estate tooling, no environment variables,
nothing external. A manual clone converges the same way by running
`python install.py`, then restart ComfyUI.

## Provenance

Published from the inference estate (`inference.cpp` repo, `plugins/comfyui/custom_nodes/melite-audio-nodes`) on 2026-09-01.

## License

MIT — see LICENSE.
