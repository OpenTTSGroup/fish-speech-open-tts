# fish-speech-open-tts

**English** · [中文](./README.zh.md)

An [Open TTS](https://github.com/OpenTTSGroup/open-tts-spec)–compliant,
OpenAI-compatible HTTP wrapper around [Fish Speech (S2-Pro)](https://huggingface.co/fishaudio/s2-pro).
Ships as a single CUDA container published on GHCR.

## Features

- OpenAI-compatible `POST /v1/audio/speech` — any OpenAI SDK works out of the box.
- File-system-driven voice library: drop `alice.wav` + `alice.txt` into the mount
  and it's immediately callable as `voice="file://alice"`.
- Streaming synthesis over `Transfer-Encoding: chunked` (`POST /v1/audio/realtime`).
- One-shot clone uploads via `POST /v1/audio/clone` (multipart) — no persistence.
- Weight-only int8 quantization for the 4B LLaMA backbone (CUDA only).
- Configurable `max_seq_len`, warm-up token budget, and torch.compile toggle.

## Quick Start (Docker)

```bash
mkdir -p voices cache
# Prepare a reference voice pair:
cp ~/my-voice.wav voices/alice.wav
echo "This is the transcript of the reference clip." > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/fish-speech-open-tts:latest
```

The first boot downloads ~8 GB of weights from HuggingFace into the mounted
`cache/` directory. `GET /healthz` returns `"status": "loading"` until the
model is ready; afterwards it flips to `"ok"`.

Call the service with any OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.audio.speech.create(
    model="any",
    voice="file://alice",
    input="Hello, world!",
    response_format="mp3",
)
resp.stream_to_file("out.mp3")
```

## Configuration

All configuration is via environment variables.

### Engine (`FISHSPEECH_*`)

| Variable | Default | Description |
|---|---|---|
| `FISHSPEECH_MODEL` | `fishaudio/s2-pro` | HuggingFace repo id (downloaded via `huggingface_hub`), or an existing local directory containing `model.pth` and `codec.pth`. |
| `FISHSPEECH_DEVICE` | `auto` | `auto` / `cuda` / `cpu`. |
| `FISHSPEECH_CUDA_INDEX` | `0` | GPU index for multi-GPU hosts. |
| `FISHSPEECH_DTYPE` | `bfloat16` | `float16` / `bfloat16` / `float32`. |
| `FISHSPEECH_DECODER_CONFIG` | `modded_dac_vq` | Hydra config for the DAC decoder. |
| `FISHSPEECH_COMPILE` | `false` | `torch.compile` the LLaMA forward. Slower first call, faster steady state. |
| `FISHSPEECH_QUANTIZATION` | `none` | `none` or `int8` — weight-only quantization (CUDA only; CPU silently falls back to `none`). Quantized checkpoints are materialized once under `$HF_HOME/../quantized/<slug>-int8/` (default `/root/.cache/quantized/…`) and reused across restarts. |
| `FISHSPEECH_MAX_SEQ_LEN` | `2560` | Override the checkpoint's `max_seq_len`. `0` keeps the default. |
| `FISHSPEECH_WARMUP_TOKENS` | `128` | Warm-up target length at startup. `0` skips warm-up. |

### Service

| Variable | Default | Description |
|---|---|---|
| `VOICES_DIR` | `/voices` | Reference voices mount point. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8000` | Bind port. |
| `LOG_LEVEL` | `info` | Uvicorn log level. |
| `MAX_INPUT_CHARS` | `8000` | Text length ceiling. |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | Fallback when the request omits `response_format`. |
| `MAX_CONCURRENCY` | `1` | In-flight inference ceiling. |
| `MAX_QUEUE_SIZE` | `0` | Queue ceiling; `0` = unlimited. |
| `QUEUE_TIMEOUT` | `0` | Queue wait timeout in seconds; `0` = unlimited. |
| `MAX_AUDIO_BYTES` | `20971520` | Upload ceiling for `POST /v1/audio/clone`. |
| `CORS_ENABLED` | `false` | `true` opens all endpoints to any origin. |

## API Reference

| Method | Path | Summary |
|---|---|---|
| GET | `/healthz` | Engine status, capability matrix, concurrency snapshot. |
| GET | `/v1/audio/voices` | List available voices. |
| GET | `/v1/audio/voices/preview?id=<id>` | Serve the reference WAV of a file voice. |
| POST | `/v1/audio/speech` | OpenAI-compatible one-shot synthesis. |
| POST | `/v1/audio/clone` | Multipart upload + synthesis in a single request. |
| POST | `/v1/audio/realtime` | Chunked streaming synthesis. |

### Capabilities matrix

```json
{
  "clone": true,
  "streaming": true,
  "design": false,
  "languages": false,
  "builtin_voices": false
}
```

`/v1/audio/design` and `/v1/audio/languages` are **not** exposed (the engine
requires a reference clip and does not publish an explicit language list).

### `POST /v1/audio/speech`

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `input` | string | — | **required** | Text to synthesize. 1..`MAX_INPUT_CHARS`. |
| `voice` | string | — | **required** | Must be `"file://<id>"` — fish-speech has no built-in voices. |
| `response_format` | enum | `mp3` | **supported** | `mp3` / `opus` / `aac` / `flac` / `wav` / `pcm`. |
| `model` | string | `null` | **ignored** | Accepted for OpenAI compatibility. |
| `speed` | float | `1.0` | **ignored** | Accepted for OpenAI compatibility; fish-speech has no speed knob. |
| `instructions` | string | `null` | **ignored** | Accepted for OpenAI compatibility; fish-speech has no instruct API. |

### `POST /v1/audio/clone`

`multipart/form-data`:

| Field | Type | Default | Status | Description |
|---|---|---|---|---|
| `audio` | file | — | **required** | Reference audio. `.wav` / `.mp3` / `.flac` / `.ogg` / `.opus` / `.m4a` / `.aac` / `.webm`, up to `MAX_AUDIO_BYTES`. |
| `prompt_text` | string | — | **required** | Transcript of the reference audio. |
| `input` | string | — | **required** | Text to synthesize. |
| `response_format` | string | `mp3` | **supported** | See §3.1 of the spec. |
| `speed` | float | `1.0` | **ignored** | |
| `instructions` | string | `null` | **ignored** | |
| `model` | string | `null` | **ignored** | |

### `POST /v1/audio/realtime`

Fields identical to `/v1/audio/speech`. `response_format` must be one of
`mp3` / `pcm` / `opus` / `aac` — `wav` / `flac` return `422`.

## Voice Directory

Drop a `<id>.wav` + `<id>.txt` pair (and optionally `<id>.yml` with metadata)
into the directory mounted at `$VOICES_DIR`. Addressable as `voice="file://<id>"`.
Example `alice.yml`:

```yaml
name: "Alice"
gender: female
age: adult
language: en
accent: british
tags: [warm, calm]
```

## Development

```bash
git clone --recurse-submodules https://github.com/<owner>/fish-speech-open-tts
cd fish-speech-open-tts
docker build -f docker/Dockerfile -t fish-speech-open-tts:local .
```

Run tests against a local instance with `curl`:

```bash
curl -s http://localhost:8000/healthz | jq
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello world","voice":"file://alice","response_format":"mp3"}' \
  --output out.mp3
```

See the [Open TTS specification](https://github.com/OpenTTSGroup/open-tts-spec)
for protocol details.
