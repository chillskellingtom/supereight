ARG BASE_IMAGE=intel/intel-optimized-pytorch:latest
FROM ${BASE_IMAGE}

WORKDIR /workspace

# System deps (ffmpeg for whisper/scenedetect)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
# Attempt to install Intel XPU-enabled torch/IPEX (best-effort; falls back to CPU if unavailable)
RUN --mount=type=cache,target=/root/.cache/pip \
    (pip install --no-cache-dir torch==2.3.0a0+xpu intel-extension-for-pytorch==2.3.0 -f https://developer.intel.com/ipex-whl-stable-xpu || true)

# Project code
COPY . /workspace
RUN chmod +x /workspace/scripts/docker_entrypoint.sh /workspace/scripts/run_pipeline.sh
# Prefetch models to /models for offline runs
RUN mkdir -p /models && PYTHONPATH=/workspace python /workspace/scripts/prefetch_models.py || true

# Default env for Intel GPUs and logging
ENV ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \
    PYTHONUNBUFFERED=1 \
    SUPEREIGHT_INPUT=/data/inputs \
    SUPEREIGHT_OUTPUT=/data/processed \
    SUPEREIGHT_SCENES_FOLDER=/data/processed/scenes \
    SUPEREIGHT_MODEL_CACHE=/models

ENTRYPOINT ["/workspace/scripts/docker_entrypoint.sh"]
