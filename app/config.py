from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Engine (FISHSPEECH_* prefix) ----------------------------------------
    fishspeech_model: str = Field(
        default="fishaudio/s2-pro",
        description=(
            "HuggingFace repo id (downloaded via huggingface_hub) or an "
            "existing local directory containing model.pth + codec.pth."
        ),
    )
    fishspeech_device: Literal["auto", "cuda", "cpu"] = "auto"
    fishspeech_cuda_index: int = Field(default=0, ge=0)
    fishspeech_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Compute precision. bfloat16 is fish-speech's reference setting.",
    )
    fishspeech_decoder_config: str = Field(
        default="modded_dac_vq",
        description="Hydra config name for the DAC decoder (fish_speech.configs).",
    )
    fishspeech_compile: bool = Field(
        default=False,
        description="torch.compile the LLaMA forward. Slower first call, faster steady state.",
    )
    fishspeech_quantization: Literal["none", "int8"] = Field(
        default="none",
        description="Weight-only quantization for the LLaMA checkpoint. CUDA only; CPU falls back to 'none'.",
    )
    fishspeech_reference_cache_size: int = Field(
        default=16,
        ge=1,
        description="Maximum number of cached reference voices (LRU).",
    )
    fishspeech_warmup_tokens: int = Field(
        default=128,
        ge=0,
        le=16384,
        description="Number of tokens to generate for the startup warm-up pass. 0 skips warm-up.",
    )
    fishspeech_max_seq_len: int = Field(
        default=2560,
        ge=0,
        description="Override model.config.max_seq_len. 0 keeps the checkpoint's default.",
    )

    # Generation hyperparameters (internal defaults; not part of the public API).
    fishspeech_max_new_tokens: int = Field(default=1024, ge=1, le=16384)
    fishspeech_chunk_length: int = Field(default=200, ge=0, le=1000)
    fishspeech_top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    fishspeech_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    fishspeech_repetition_penalty: float = Field(default=1.2, ge=0.5, le=2.0)

    # --- Service-level (no prefix) -------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(default=8000, ge=1)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = "mp3"
    max_concurrency: int = Field(default=1, ge=1)
    max_queue_size: int = Field(default=0, ge=0)
    queue_timeout: float = Field(default=0.0, ge=0.0)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    cors_enabled: bool = False

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)

    @property
    def resolved_device(self) -> str:
        if self.fishspeech_device == "cpu":
            return "cpu"
        if self.fishspeech_device == "cuda":
            return f"cuda:{self.fishspeech_cuda_index}"
        import torch

        if torch.cuda.is_available():
            return f"cuda:{self.fishspeech_cuda_index}"
        return "cpu"

    @property
    def torch_precision(self):
        import torch

        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.fishspeech_dtype]

    @property
    def effective_quantization(self) -> Literal["none", "int8"]:
        """Quantization only applies on CUDA; on CPU we silently fall back to none."""
        if self.fishspeech_quantization != "none" and not self.resolved_device.startswith("cuda"):
            return "none"
        return self.fishspeech_quantization


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
