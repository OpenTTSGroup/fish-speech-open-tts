from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class Capabilities(BaseModel):
    clone: bool = Field(description="Zero-shot cloning support.")
    streaming: bool = Field(description="Chunked realtime synthesis support.")
    design: bool = Field(description="Text-only voice design support.")
    languages: bool = Field(description="Explicit language list support.")
    builtin_voices: bool = Field(description="Engine ships built-in voices.")


class ConcurrencySnapshot(BaseModel):
    max: int = Field(description="Global concurrency ceiling.")
    active: int = Field(description="Currently in-flight synthesis jobs.")
    queued: int = Field(description="Waiters blocked on the semaphore.")


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"] = Field(
        description="Engine readiness state."
    )
    model: str = Field(description="Loaded model identifier.")
    sample_rate: int = Field(description="Inference output sample rate (Hz).")
    capabilities: Capabilities = Field(description="Discovered engine capabilities.")
    device: Optional[str] = Field(default=None, description='e.g. "cuda:0" or "cpu".')
    dtype: Optional[str] = Field(default=None, description='e.g. "float16".')
    concurrency: Optional[ConcurrencySnapshot] = Field(
        default=None, description="Live concurrency snapshot."
    )


class VoiceInfo(BaseModel):
    id: str = Field(
        description='Voice identifier. "file://<name>" for disk voices, raw name for built-ins.'
    )
    preview_url: Optional[str] = Field(
        description="Preview URL for file voices; null for built-ins."
    )
    prompt_text: Optional[str] = Field(
        description="Reference transcript for file voices; null for built-ins."
    )
    metadata: Optional[dict[str, Any]] = Field(
        description="Optional metadata dict from <id>.yml."
    )


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo] = Field(description="Discovered voices.")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = Field(
        default=None,
        description="Accepted for OpenAI compatibility; ignored.",
    )
    input: str = Field(
        min_length=1,
        description="Text to synthesize.",
    )
    voice: str = Field(
        description='Built-in voice name, or "file://<id>" for a disk reference.'
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Output container/codec; defaults to the service setting.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback rate; accepted for OpenAI compatibility, ignored by fish-speech.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Style hint; accepted for OpenAI compatibility, ignored by fish-speech.",
    )


RealtimeRequest = SpeechRequest
