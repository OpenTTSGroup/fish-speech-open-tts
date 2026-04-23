#!/usr/bin/env bash
set -euo pipefail

# Engine defaults
: "${FISHSPEECH_MODEL:=fishaudio/s2-pro}"
: "${FISHSPEECH_DEVICE:=auto}"
: "${FISHSPEECH_DTYPE:=bfloat16}"
: "${FISHSPEECH_DECODER_CONFIG:=modded_dac_vq}"
: "${FISHSPEECH_COMPILE:=false}"
: "${FISHSPEECH_QUANTIZATION:=none}"
: "${FISHSPEECH_WARMUP_TOKENS:=128}"
: "${FISHSPEECH_MAX_SEQ_LEN:=2560}"

# Service-level defaults
: "${VOICES_DIR:=/voices}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${CORS_ENABLED:=false}"
: "${PYTHONPATH:=/opt/api:/opt/api/engine}"
: "${FISHSPEECH_ROOT:=/opt/api/engine}"

export FISHSPEECH_MODEL FISHSPEECH_DEVICE FISHSPEECH_DTYPE \
       FISHSPEECH_DECODER_CONFIG FISHSPEECH_COMPILE FISHSPEECH_QUANTIZATION \
       FISHSPEECH_WARMUP_TOKENS FISHSPEECH_MAX_SEQ_LEN \
       VOICES_DIR HOST PORT LOG_LEVEL CORS_ENABLED \
       PYTHONPATH FISHSPEECH_ROOT

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
