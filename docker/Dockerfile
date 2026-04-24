# syntax=docker/dockerfile:1.7

# ============================================================================
# Build arguments
#   CUDA_VERSION    — base image CUDA version. fish-speech pins torch==2.8.0,
#                     so the default targets the cu128 wheel (12.8.0).
#   UBUNTU_VERSION  — base OS suffix.
#   TORCH_VERSION   — upstream fish-speech requirement.
#   TORCH_CUDA_TAG  — PyTorch wheel index tag matching CUDA_VERSION.
# ============================================================================
ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.8.0
ARG TORCH_CUDA_TAG=cu128

# ============================================================================
# Builder: python venv with torch, fish-speech, and the API layer.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS builder

ARG TORCH_VERSION
ARG TORCH_CUDA_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# build-essential + portaudio19-dev are needed because fish-speech pulls in
# pyaudio (an sdist that builds a C extension at install time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
        portaudio19-dev \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip

# PyTorch — matched to fish-speech's torch==2.8.0 pin.
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

# API-layer dependencies (FastAPI / pydantic / soundfile / av / pyyaml).
COPY docker/requirements.api.txt /tmp/requirements.api.txt
RUN pip install -r /tmp/requirements.api.txt

# Upstream fish-speech — installs transformers, lightning, descript-audio-codec,
# and every other runtime dep declared in engine/pyproject.toml. We copy only
# the submodule tree (without its .git metadata) to keep the build context tight.
COPY engine /build/engine
RUN pip install /build/engine

# Extra engine-layer packages (HuggingFace download accelerators etc.).
COPY docker/requirements.engine.txt /tmp/requirements.engine.txt
RUN pip install -r /tmp/requirements.engine.txt

# Strip bytecode caches left behind by eager imports during install.
RUN find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Runtime: minimal image carrying only the venv + fish-speech source + app/.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/api:/opt/api/engine \
    FISHSPEECH_ROOT=/opt/api/engine

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        gcc \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/api

# fish-speech is already pip-installed into the venv, but the submodule also
# carries runtime assets (hydra configs under fish_speech/configs/, tokenizer
# resources, tools/llama/quantize.py) that are imported at runtime and live
# under the engine tree. Keep a copy at /opt/api/engine for import-by-path.
COPY engine /opt/api/engine
COPY app /opt/api/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/voices", "/root/.cache"]
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
