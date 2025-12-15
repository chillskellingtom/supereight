ARG BASE_IMAGE=intel/intel-optimized-pytorch:xpu-pytorch-2.3.0-jupyter
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

# Project code
COPY . /workspace
RUN chmod +x /workspace/scripts/docker_entrypoint.sh /workspace/scripts/run_pipeline.sh

# Default env for Intel GPUs and logging
ENV ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \
    PYTHONUNBUFFERED=1 \
    SUPEREIGHT_INPUT=/data/inputs \
    SUPEREIGHT_OUTPUT=/data/processed \
    SUPEREIGHT_SCENES_FOLDER=/data/processed/scenes

ENTRYPOINT ["/workspace/scripts/docker_entrypoint.sh"]
