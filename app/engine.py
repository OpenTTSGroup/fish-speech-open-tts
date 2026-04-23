from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

import numpy as np

from app.config import Settings


log = logging.getLogger(__name__)

# Fish-speech's ``tools/`` package lives inside the engine submodule; when the
# upstream is pip-installed it is already importable, but local runs rely on
# PYTHONPATH. We honour ``FISHSPEECH_ROOT`` as a fallback for dev setups.
_fs_root = os.environ.get("FISHSPEECH_ROOT")
if _fs_root and _fs_root not in sys.path:
    sys.path.insert(0, _fs_root)


def _resolve_model_dir(repo_or_path: str) -> Path:
    """Resolve ``repo_or_path`` to an on-disk checkpoint directory.

    If the value is an existing directory, use it as-is (user-mounted
    checkpoint). Otherwise treat it as a HuggingFace repo id and let
    ``huggingface_hub`` download / cache it.
    """
    p = Path(repo_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_or_path))


def _quant_cache_root() -> Path:
    """Quantised-checkpoint cache, kept under the persistent ``/root/.cache``
    volume but outside the HuggingFace hub tree so we don't pollute
    ``huggingface_hub``'s own bookkeeping.

    Defaults to ``/root/.cache/quantized`` and follows ``HF_HOME`` when the
    user has relocated the cache root.
    """
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    return hf_home.parent / "quantized"


def _slugify_model_id(model_id: str) -> str:
    """Make a path-safe slug out of a HuggingFace repo id or local path."""
    return (
        model_id.replace("\\", "--")
        .replace("/", "--")
        .replace(":", "--")
        .strip("-")
    )


def _prepare_quantized_checkpoint(
    model_dir: Path,
    model_id: str,
    mode: Literal["int8"],
) -> Path:
    """Return the path of a weight-only int8 quantised checkpoint, producing
    it on the fly under :func:`_quant_cache_root` when absent.

    fish-speech's LLaMA loader (``BaseTransformer.from_pretrained``) decides
    the quantisation scheme purely from *the path name*: it looks for the
    substring ``int8``. We therefore name our artefact ``<slug>-int8``.
    """
    assert mode == "int8", f"unsupported quantisation mode: {mode}"
    dst = _quant_cache_root() / f"{_slugify_model_id(model_id)}-int8"
    model_pth = dst / "model.pth"
    if dst.is_dir() and model_pth.exists():
        log.info("reusing cached quantised checkpoint at %s", dst)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    log.info("quantising checkpoint to %s (mode=int8)", dst)

    import torch
    from fish_speech.models.text2semantic.inference import init_model
    from tools.llama.quantize import WeightOnlyInt8QuantHandler

    base_model, _ = init_model(
        checkpoint_path=model_dir,
        device="cpu",
        precision=torch.bfloat16,
        compile=False,
    )
    quant_state = WeightOnlyInt8QuantHandler(base_model).create_quantized_state_dict()

    # Copy config + tokenizer assets, strip weights; then drop the quantised
    # state dict as model.pth.
    dst_tmp = dst.with_name(dst.name + ".tmp")
    if dst_tmp.exists():
        shutil.rmtree(dst_tmp)
    shutil.copytree(
        model_dir,
        dst_tmp,
        ignore=shutil.ignore_patterns(
            "model.pth",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ),
    )
    # codec.pth belongs to the decoder and is loaded separately; keep it where
    # it already lives (in the base model dir). The quantised copy only needs
    # the LLaMA weights.
    codec_in_tmp = dst_tmp / "codec.pth"
    if codec_in_tmp.exists():
        codec_in_tmp.unlink()
    torch.save(quant_state, dst_tmp / "model.pth")
    dst_tmp.rename(dst)

    # Help the gc release the fp32/bf16 state before the real model loads.
    del base_model, quant_state
    return dst


def _install_max_seq_len_patch(max_seq_len: int) -> None:
    """Monkey-patch ``init_model`` so the caller-provided ``max_seq_len``
    overrides the checkpoint default.

    We don't carry a fork of ``init_model`` because the upstream function
    also wires up ``decode_one_token`` / compile, but we do need to inject
    ``max_length`` into ``DualARTransformer.from_pretrained``. The cheapest
    sound intervention is to replace ``from_pretrained`` with a wrapper
    that fills in ``max_length`` when the caller didn't.
    """
    if max_seq_len <= 0:
        return

    from fish_speech.models.text2semantic import llama as _llama

    original = _llama.BaseTransformer.from_pretrained
    if getattr(original, "_open_tts_patched", False):
        return

    @staticmethod
    def _patched(path, load_weights=False, max_length=None, lora_config=None, rope_base=None):
        effective = max_length if max_length is not None else max_seq_len
        return original(
            path,
            load_weights=load_weights,
            max_length=effective,
            lora_config=lora_config,
            rope_base=rope_base,
        )

    _patched._open_tts_patched = True  # type: ignore[attr-defined]
    _llama.BaseTransformer.from_pretrained = _patched  # type: ignore[assignment]


class TTSEngine:
    """Thin async wrapper around fish-speech's TTSInferenceEngine."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._device = settings.resolved_device
        self._precision = settings.torch_precision
        self._dtype_str = settings.fishspeech_dtype

        _install_max_seq_len_patch(settings.fishspeech_max_seq_len)

        base_model_dir = _resolve_model_dir(settings.fishspeech_model)

        quant = settings.effective_quantization
        if quant == "none":
            llama_dir = base_model_dir
        else:
            llama_dir = _prepare_quantized_checkpoint(
                base_model_dir,
                settings.fishspeech_model,
                quant,
            )

        codec_ckpt = base_model_dir / "codec.pth"
        if not codec_ckpt.exists():
            raise FileNotFoundError(
                f"codec.pth not found under {base_model_dir}; "
                "ensure the fish-speech checkpoint is complete."
            )

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        log.info("loading fish-speech LLaMA from %s (compile=%s, quant=%s, max_seq_len=%s)",
                 llama_dir, settings.fishspeech_compile, quant, settings.fishspeech_max_seq_len)
        self._llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_dir,
            device=self._device,
            precision=self._precision,
            compile=settings.fishspeech_compile,
        )
        log.info("loading fish-speech DAC decoder from %s", codec_ckpt)
        self._decoder = load_decoder_model(
            config_name=settings.fishspeech_decoder_config,
            checkpoint_path=str(codec_ckpt),
            device=self._device,
        )

        self._engine = TTSInferenceEngine(
            llama_queue=self._llama_queue,
            decoder_model=self._decoder,
            precision=self._precision,
            compile=settings.fishspeech_compile,
        )

        self._sample_rate = int(self._decoder.sample_rate)
        self._ref_lock = threading.Lock()

        self._maybe_warm_up(settings.fishspeech_warmup_tokens)

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_str(self) -> str:
        return self._dtype_str

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_id(self) -> str:
        return self._settings.fishspeech_model

    @property
    def builtin_voices_list(self) -> list[str]:
        # fish-speech has no built-in speaker catalogue; every voice is a
        # zero-shot clone driven by the supplied reference audio.
        return []

    # ------------------------------------------------------------------
    # Warm-up

    def _maybe_warm_up(self, tokens: int) -> None:
        if tokens <= 0:
            log.info("warm-up skipped (FISHSPEECH_WARMUP_TOKENS=0)")
            return
        from fish_speech.utils.schema import ServeTTSRequest

        log.info("warming up fish-speech (target max_new_tokens=%d)", tokens)
        req = ServeTTSRequest(
            text="Warm up.",
            references=[],
            reference_id=None,
            max_new_tokens=tokens,
            chunk_length=self._settings.fishspeech_chunk_length or 200,
            top_p=self._settings.fishspeech_top_p,
            repetition_penalty=self._settings.fishspeech_repetition_penalty,
            temperature=self._settings.fishspeech_temperature,
            streaming=False,
            format="wav",
            normalize=True,
        )
        try:
            for _ in self._engine.inference(req):
                pass
            log.info("warm-up complete")
        except Exception:  # pragma: no cover - surfaced via logs
            log.exception("warm-up failed (continuing; first real request may be slow)")

    # ------------------------------------------------------------------
    # Request construction

    def _build_request(
        self,
        text: str,
        *,
        ref_audio: Optional[str],
        ref_text: Optional[str],
        streaming: bool,
    ):
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        refs = []
        if ref_audio is not None:
            audio_bytes = Path(ref_audio).read_bytes()
            refs.append(
                ServeReferenceAudio(audio=audio_bytes, text=ref_text or "")
            )

        return ServeTTSRequest(
            text=text,
            references=refs,
            reference_id=None,
            max_new_tokens=self._settings.fishspeech_max_new_tokens,
            chunk_length=self._settings.fishspeech_chunk_length or 200,
            top_p=self._settings.fishspeech_top_p,
            repetition_penalty=self._settings.fishspeech_repetition_penalty,
            temperature=self._settings.fishspeech_temperature,
            streaming=streaming,
            format="wav",
            normalize=True,
            use_memory_cache="on" if ref_audio is not None else "off",
        )

    @staticmethod
    def _as_float32(arr) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        if out.ndim > 1:
            out = out.reshape(-1)
        return out

    # ------------------------------------------------------------------
    # Non-streaming synthesis

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float] = None,  # kept for spec parity; not used
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **_: object,
    ) -> np.ndarray:
        if instructions:
            log.warning("instructions ignored: fish-speech has no instruct API")
        if speed != 1.0:
            log.warning("speed != 1.0 ignored: fish-speech does not expose a speed knob")

        req = self._build_request(
            text, ref_audio=ref_audio, ref_text=ref_text, streaming=False
        )

        def _run() -> np.ndarray:
            final_audio: Optional[np.ndarray] = None
            for result in self._engine.inference(req):
                if result.code == "error":
                    raise RuntimeError(
                        str(result.error) if result.error else "inference failed"
                    )
                if result.code == "final" and isinstance(result.audio, tuple):
                    final_audio = self._as_float32(result.audio[1])
            if final_audio is None:
                raise RuntimeError("no audio generated")
            return final_audio

        # fish-speech's inference is synchronous and blocks on an internal
        # queue; offloading keeps the event loop responsive.
        return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # Streaming synthesis

    async def synthesize_realtime(
        self,
        text: str,
        *,
        voice: str,  # unused: voice resolution happens in the server layer
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_mtime: Optional[float] = None,  # unused
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **_: object,
    ) -> AsyncIterator[np.ndarray]:
        if instructions:
            log.warning("instructions ignored in stream: fish-speech has no instruct API")
        if speed != 1.0:
            log.warning("speed != 1.0 ignored in stream")

        req = self._build_request(
            text, ref_audio=ref_audio, ref_text=ref_text, streaming=True
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        sentinel = object()

        def _producer() -> None:
            try:
                for result in self._engine.inference(req):
                    if result.code == "error":
                        raise RuntimeError(
                            str(result.error) if result.error else "inference failed"
                        )
                    if result.code == "segment" and isinstance(result.audio, tuple):
                        arr = self._as_float32(result.audio[1])
                        if arr.size:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(arr), loop
                            ).result()
                    # 'header' and 'final' are intentionally skipped — header
                    # would inject a WAV preamble and 'final' is the full
                    # concatenation we've already streamed chunk-by-chunk.
            except Exception as exc:  # pragma: no cover - surfaced via stream
                log.exception("streaming producer failed")
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
                except Exception:
                    pass
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()
                except Exception:
                    pass

        thread = threading.Thread(
            target=_producer, name="fishspeech-stream", daemon=True
        )
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
