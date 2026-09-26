"""moss_sfx_v2 — pure-torch sound-effect engine (TTS-Audio-Suite pattern).

No C++, no GGUF, no ctypes, no eviction nodes. The official v2 diffusion
pipeline (MossSoundEffectPipeline, vendored in ``moss_sfx_impl/``) loads the
HF checkpoint directly via ``from_pretrained()``; ComfyUI manages the model
lifecycle like any torch model — dropping references + gc.collect() +
torch.cuda.empty_cache() return all VRAM.

The DiT is torch.compile'd (vendored diffsynth); inductor artifacts persist
under the compile cache so the FIRST generation is slow (measured 59.8s for
20 steps incl. compile) and later ones are fast (~0.5s/step warm).

History (2026-08-11): this family previously ran the audiocpp C++ fork
(ggml/GUFF via ctypes). Its DiTRunner leaked the 5.4 GB migrated weight
buffer every session destroy — "first gen worked, second crashed". The
torch pipeline has no such class of bug: memory is plain torch tensors.
The C++ path remains for ace_step music until its torch port exists.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

logger = logging.getLogger("audiocore.moss_sfx_v2")

# The vendored pipeline package (MossSoundEffectPipeline + diffsynth).
_IMPL_DIR = Path(__file__).resolve().parent / "moss_sfx_impl"

# Official defaults for the diffusion loop (their engine node's tooltips).
_DEFAULT_CFG_SCALE = 5.0
_DEFAULT_INFERENCE_STEPS = 50
_DEFAULT_SECONDS = 10.0
_SIGMA_SHIFT = 5.0


class TorchSfxEngine:
    """MOSS-SFX v2 as a plain torch diffusion pipeline.

    Lifecycle mirrors TTS-Audio-Suite's MossSoundEffectV2Engine: lazy load at
    first generate; unload() = pipeline.to("cpu") → None → gc → empty_cache.
    NO ``current_loaded_models`` registration, NO eviction node — the pipeline
    is a normal Python object. ComfyUI's node-instance cache holds it between
    generations (no cold restarts); gc + empty_cache return VRAM when it is
    dropped.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)
        self.pipeline: Any = None
        self._compile_notice_shown = False
        self._configure_compile_cache()

    # ── Persistent torch.compile cache (first gen slow, then fast) ──────

    def _configure_compile_cache(self) -> Optional[Path]:
        """Enable persistent Inductor graph reuse.

        The durable location (the models dir's sibling — TTS-Audio-Suite's
        convention) is used when writable. In the container the models dir is
        read-only, so torch's own default cache (~/.cache or /tmp/
        torchinductor_*, warmed by earlier generations) is left in place —
        same persistence, already warm.
        """
        try:
            model_dir = Path(self.model_path).resolve()
            cache_root = model_dir.parent.parent / "compile_cache"
            try:
                cache_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.info(
                    "compile cache %s not writable — using torch defaults",
                    cache_root,
                )
                return None
            inductor_dir = cache_root / "torchinductor"
            triton_dir = cache_root / "triton"
            inductor_dir.mkdir(parents=True, exist_ok=True)
            triton_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
            os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
            return cache_root
        except Exception as exc:  # pragma: no cover — env/filesystem edge
            logger.warning("moss_sfx_v2: compile cache setup failed: %s", exc)
            return None

    # ── Load ─────────────────────────────────────────────────────────────

    def load(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Load the pipeline from the HF checkpoint dir (lazy, once)."""
        if self.pipeline is not None:
            return True
        try:
            if str(_IMPL_DIR) not in sys.path:
                sys.path.insert(0, str(_IMPL_DIR))
            from moss_soundeffect_v2 import MossSoundEffectPipeline  # noqa: PLC0415

            print("🔄 Loading MOSS-SFX v2 (torch pipeline)", flush=True)
            t0 = time.perf_counter()
            load_options = {
                "device": "cuda",
                "torch_dtype": torch.bfloat16,
                "local_files_only": True,
            }
            if on_progress is not None:
                load_options["on_progress"] = on_progress
            self.pipeline = MossSoundEffectPipeline.from_pretrained(
                self.model_path,
                **load_options,
            )
            print(
                f"✅ MOSS-SFX v2 loaded in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            return True
        except Exception as exc:
            logger.error("moss_sfx_v2 torch load failed: %s", exc)
            print(f"❌ moss_sfx_v2 torch load failed: {exc}", flush=True)
            self.pipeline = None
            return False

    # ── Generate ─────────────────────────────────────────────────────────

    @staticmethod
    def _progress_bar_cmd(
        on_progress: Optional[Callable[[int, int], None]],
    ) -> Callable[[Iterable], Iterable]:
        """Wrap the pipeline's progress iterable: tqdm + ComfyUI progress.

        Also honors ComfyUI cancellation via
        ``throw_exception_if_processing_interrupted`` (checked each step).
        """
        from tqdm.auto import tqdm  # noqa: PLC0415

        def _wrap(iterable: Iterable) -> Iterable:
            total = len(iterable) if hasattr(iterable, "__len__") else None
            bar = tqdm(iterable, total=total, desc="MOSS-SFX v2", dynamic_ncols=True)
            for step, item in enumerate(bar, start=1):
                if on_progress is not None and total:
                    on_progress(step, total)
                try:
                    import comfy.model_management as model_management  # noqa: PLC0415

                    checker = getattr(
                        model_management,
                        "throw_exception_if_processing_interrupted",
                        None,
                    )
                    if callable(checker):
                        checker()
                except ImportError:
                    pass
                yield item

        return _wrap

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        guidance_scale: float = _DEFAULT_CFG_SCALE,
        num_inference_steps: int = _DEFAULT_INFERENCE_STEPS,
        duration_seconds: float = _DEFAULT_SECONDS,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[list[float], int]:
        """Generate a sound effect. Returns (pcm_float32, sample_rate).

        Contract-identical to the C++ session's speech output so the
        AudiocoreTTS node surface never changes.
        """
        if not self.load():
            raise RuntimeError(
                f"Failed to load moss_sfx_v2 from {self.model_path}. "
                "No silent fallback — see ComfyUI logs for the per-node error."
            )
        if not self._compile_notice_shown:
            print(
                "⚙️ MOSS-SFX v2: first generation compiles the DiT "
                "(can take minutes); artifacts persist in the compile cache "
                "— later generations are fast.",
                flush=True,
            )
            self._compile_notice_shown = True
        waveform = self.pipeline(
            prompt=str(prompt),
            seconds=float(duration_seconds),
            num_inference_steps=int(num_inference_steps),
            cfg_scale=float(guidance_scale),
            sigma_shift=_SIGMA_SHIFT,
            negative_prompt="",
            seed=int(seed),
            progress_bar_cmd=self._progress_bar_cmd(on_progress),
        )
        w = waveform.detach().cpu().float()
        return w.reshape(-1).tolist(), int(self.pipeline.sample_rate)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def unload(self) -> None:
        """Drop the pipeline; gc + empty_cache return all VRAM."""
        if self.pipeline is None:
            return
        try:
            self.pipeline.to("cpu")
        except Exception:
            pass
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def loaded_bytes(self) -> int:
        """Total parameter bytes of the pipeline (for the info node)."""
        if self.pipeline is None:
            return 0
        engine = getattr(self.pipeline, "engine", None)
        mod = engine if engine is not None else self.pipeline
        total = 0
        for p in mod.parameters():
            total += p.numel() * p.element_size()
        return total


__all__ = ["TorchSfxEngine"]
