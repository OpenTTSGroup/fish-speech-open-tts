from __future__ import annotations

import io
from typing import Optional

import numpy as np


CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}

STREAMABLE_FORMATS: frozenset[str] = frozenset({"mp3", "pcm", "opus", "aac"})

_PYAV_CONTAINER_FORMAT = {"mp3": "mp3", "opus": "ogg", "aac": "adts"}
_PYAV_CODEC = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}


def _normalize(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False)
    np.clip(arr, -1.0, 1.0, out=arr)
    return arr


def _to_pcm16_bytes(samples: np.ndarray) -> bytes:
    scaled = np.clip(samples * 32767.0, -32768.0, 32767.0)
    return scaled.astype("<i2", copy=False).tobytes()


def _encode_soundfile(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    elif fmt == "flac":
        sf.write(buf, samples, sample_rate, format="FLAC")
    else:  # pragma: no cover
        raise ValueError(f"soundfile cannot encode {fmt}")
    return buf.getvalue()


def _encode_pyav(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=_PYAV_CONTAINER_FORMAT[fmt])
    try:
        stream = container.add_stream(_PYAV_CODEC[fmt], rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="flt", layout="mono"
        )
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()


def encode(samples: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    """Encode mono float32 samples into the requested container/codec."""
    if fmt not in CONTENT_TYPES:
        raise ValueError(f"unsupported response_format: {fmt}")
    arr = _normalize(samples)
    if fmt == "pcm":
        return _to_pcm16_bytes(arr), CONTENT_TYPES[fmt]
    if fmt in ("wav", "flac"):
        return _encode_soundfile(arr, sample_rate, fmt), CONTENT_TYPES[fmt]
    return _encode_pyav(arr, sample_rate, fmt), CONTENT_TYPES[fmt]


class StreamEncoder:
    """Incremental encoder driving realtime streaming responses."""

    def __init__(self, sample_rate: int, fmt: str) -> None:
        if fmt not in STREAMABLE_FORMATS:
            raise ValueError(f"streaming not supported for {fmt}")
        self._sample_rate = sample_rate
        self._fmt = fmt
        self._buf: Optional[io.BytesIO] = None
        self._container = None
        self._stream = None
        self._cursor = 0
        if fmt != "pcm":
            self._open_pyav()

    @property
    def content_type(self) -> str:
        return CONTENT_TYPES[self._fmt]

    def _open_pyav(self) -> None:
        import av

        self._buf = io.BytesIO()
        self._container = av.open(
            self._buf, mode="w", format=_PYAV_CONTAINER_FORMAT[self._fmt]
        )
        self._stream = self._container.add_stream(
            _PYAV_CODEC[self._fmt], rate=self._sample_rate
        )
        self._stream.layout = "mono"

    def _drain(self) -> bytes:
        assert self._buf is not None
        data = self._buf.getvalue()
        out = data[self._cursor :]
        self._cursor = len(data)
        return out

    def encode(self, chunk: np.ndarray) -> bytes:
        arr = _normalize(chunk)
        if arr.size == 0:
            return b""
        if self._fmt == "pcm":
            return _to_pcm16_bytes(arr)

        import av

        frame = av.AudioFrame.from_ndarray(
            arr.reshape(1, -1), format="flt", layout="mono"
        )
        frame.sample_rate = self._sample_rate
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        return self._drain()

    def flush(self) -> bytes:
        if self._fmt == "pcm":
            return b""
        if self._container is None:
            return b""
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        tail = self._drain()
        self._container = None
        self._stream = None
        return tail
