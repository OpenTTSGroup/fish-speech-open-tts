# fish-speech-open-tts

[English](./README.md) · **中文**

符合 [Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec) 的、
OpenAI 兼容的 HTTP TTS 服务，底层使用
[Fish Speech (S2-Pro)](https://huggingface.co/fishaudio/s2-pro)。
作为单一 CUDA 镜像发布到 GHCR。

## 特性

- OpenAI 兼容的 `POST /v1/audio/speech`，任何 OpenAI SDK 都能直接用。
- 文件系统驱动的声音库：把 `alice.wav` + `alice.txt` 丢到挂载目录，
  立刻就能以 `voice="file://alice"` 使用。
- 基于 `Transfer-Encoding: chunked` 的流式合成（`POST /v1/audio/realtime`）。
- `POST /v1/audio/clone` 支持一次性 multipart 上传克隆，不落盘。
- 4B LLaMA 骨干支持 weight-only int8 量化（仅 CUDA）。
- 可配置 `max_seq_len`、warm-up token 预算、以及 `torch.compile` 开关。

## 快速开始（Docker）

```bash
mkdir -p voices cache
# 准备一对参考声音文件：
cp ~/my-voice.wav voices/alice.wav
echo "参考音频的转录文本。" > voices/alice.txt

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/voices:/voices:ro" \
  -v "$PWD/cache:/root/.cache" \
  ghcr.io/openttsgroup/fish-speech-open-tts:latest
```

首次启动会从 HuggingFace 下载约 8 GB 权重到挂载的 `cache/` 目录。
`GET /healthz` 在模型就绪前返回 `"status": "loading"`，之后变为 `"ok"`。

任意 OpenAI SDK 都可直接调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.audio.speech.create(
    model="any",
    voice="file://alice",
    input="你好，世界！",
    response_format="mp3",
)
resp.stream_to_file("out.mp3")
```

## 配置

所有配置都通过环境变量完成。

### 引擎（`FISHSPEECH_*`）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FISHSPEECH_MODEL` | `fishaudio/s2-pro` | HuggingFace repo id（通过 `huggingface_hub` 下载），或已存在的本地目录（需包含 `model.pth` 和 `codec.pth`）。 |
| `FISHSPEECH_DEVICE` | `auto` | `auto` / `cuda` / `cpu`。 |
| `FISHSPEECH_CUDA_INDEX` | `0` | 多卡场景的 GPU 索引。 |
| `FISHSPEECH_DTYPE` | `bfloat16` | `float16` / `bfloat16` / `float32`。 |
| `FISHSPEECH_DECODER_CONFIG` | `modded_dac_vq` | DAC 解码器的 Hydra config。 |
| `FISHSPEECH_COMPILE` | `false` | 是否对 LLaMA forward 执行 `torch.compile`。首轮慢、稳态快。 |
| `FISHSPEECH_QUANTIZATION` | `none` | `none` 或 `int8` — weight-only 量化（仅 CUDA；CPU 静默回落 `none`）。量化产物会在首次启动时落到 `$HF_HOME/../quantized/<slug>-int8/`（默认 `/root/.cache/quantized/…`），后续启动直接复用。 |
| `FISHSPEECH_MAX_SEQ_LEN` | `2560` | 覆盖 checkpoint 的 `max_seq_len`。`0` 表示保留默认值。 |
| `FISHSPEECH_WARMUP_TOKENS` | `128` | 启动预热的目标 token 长度。`0` 跳过预热。 |

### 服务

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOICES_DIR` | `/voices` | 参考声音挂载点。 |
| `HOST` | `0.0.0.0` | 绑定地址。 |
| `PORT` | `8000` | 绑定端口。 |
| `LOG_LEVEL` | `info` | uvicorn 日志级别。 |
| `MAX_INPUT_CHARS` | `8000` | 文本长度上限。 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | 请求未指定 `response_format` 时的默认值。 |
| `MAX_CONCURRENCY` | `1` | 推理并发上限。 |
| `MAX_QUEUE_SIZE` | `0` | 排队上限；`0` = 不限。 |
| `QUEUE_TIMEOUT` | `0` | 排队等待超时（秒）；`0` = 不限。 |
| `MAX_AUDIO_BYTES` | `20971520` | `POST /v1/audio/clone` 上传上限。 |
| `CORS_ENABLED` | `false` | `true` 时对所有端点放开跨域。 |

## API 参考

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 引擎状态、能力矩阵、并发快照。 |
| GET | `/v1/audio/voices` | 列出可用声音。 |
| GET | `/v1/audio/voices/preview?id=<id>` | 返回文件克隆声音的参考 WAV。 |
| POST | `/v1/audio/speech` | OpenAI 兼容的非流式合成。 |
| POST | `/v1/audio/clone` | 一次性 multipart 上传 + 合成。 |
| POST | `/v1/audio/realtime` | 分块流式合成。 |

### 能力矩阵

```json
{
  "clone": true,
  "streaming": true,
  "design": false,
  "languages": false,
  "builtin_voices": false
}
```

`/v1/audio/design` 与 `/v1/audio/languages` **不**暴露（引擎必须有参考音频、
且不提供显式语言列表）。

### `POST /v1/audio/speech`

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `input` | string | — | **required** | 合成文本，1..`MAX_INPUT_CHARS`。 |
| `voice` | string | — | **required** | 必须为 `"file://<id>"`——fish-speech 无内置声音。 |
| `response_format` | enum | `mp3` | **supported** | `mp3` / `opus` / `aac` / `flac` / `wav` / `pcm`。 |
| `model` | string | `null` | **ignored** | OpenAI 兼容字段，服务端接受但忽略。 |
| `speed` | float | `1.0` | **ignored** | OpenAI 兼容字段；fish-speech 没有速度开关。 |
| `instructions` | string | `null` | **ignored** | OpenAI 兼容字段；fish-speech 没有 instruct API。 |

### `POST /v1/audio/clone`

`multipart/form-data`：

| 字段 | 类型 | 默认 | 状态 | 说明 |
|---|---|---|---|---|
| `audio` | file | — | **required** | 参考音频。支持 `.wav` / `.mp3` / `.flac` / `.ogg` / `.opus` / `.m4a` / `.aac` / `.webm`，上限 `MAX_AUDIO_BYTES`。 |
| `prompt_text` | string | — | **required** | 参考音频的转录文本。 |
| `input` | string | — | **required** | 合成文本。 |
| `response_format` | string | `mp3` | **supported** | 见规范 §3.1。 |
| `speed` | float | `1.0` | **ignored** | |
| `instructions` | string | `null` | **ignored** | |
| `model` | string | `null` | **ignored** | |

### `POST /v1/audio/realtime`

字段与 `/v1/audio/speech` 完全一致。`response_format` 必须是 `mp3` / `pcm`
/ `opus` / `aac` 之一——`wav` / `flac` 返回 `422`。

## 声音目录

向挂载到 `$VOICES_DIR` 的目录里放 `<id>.wav` + `<id>.txt` 三元组（
可选加 `<id>.yml` 提供元信息）。调用时使用 `voice="file://<id>"`。
示例 `alice.yml`：

```yaml
name: "Alice"
gender: female
age: adult
language: zh
accent: mandarin
tags: [warm, calm]
```

## 开发

```bash
git clone --recurse-submodules https://github.com/<owner>/fish-speech-open-tts
cd fish-speech-open-tts
docker build -f docker/Dockerfile -t fish-speech-open-tts:local .
```

以 `curl` 跑本地 smoke test：

```bash
curl -s http://localhost:8000/healthz | jq
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，世界","voice":"file://alice","response_format":"mp3"}' \
  --output out.mp3
```

协议细节参见 [Open TTS 规范](https://github.com/OpenTTSGroup/open-tts-spec)。
