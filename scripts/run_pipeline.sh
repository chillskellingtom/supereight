#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
INPUT_DIR="${SUPEREIGHT_INPUT:-/data/inputs}"
OUTPUT_DIR="${SUPEREIGHT_OUTPUT:-/data/processed}"
SCENES_DIR="${SUPEREIGHT_SCENES_FOLDER:-${OUTPUT_DIR}/scenes}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTPUT_DIR}" "${SCENES_DIR}"

echo "=== Step 1: Scene Detection ==="
"${PYTHON_BIN}" "${ROOT}/scripts/detect_scenes.py" \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}" \
  -v

echo
echo "=== Step 2: Face Export (with auto-cluster) ==="
"${PYTHON_BIN}" "${ROOT}/process_parallel.py" \
  --task faces_export \
  --scenes-folder "${SCENES_DIR}" \
  --frame-stride 12 \
  --min-face-score 0.6 \
  --face-crop-size 256 \
  --face-max-per-track 3 \
  --auto-cluster \
  --cluster-method hierarchical \
  --cluster-similarity 0.90 \
  --cluster-min-samples 5 \
  --cluster-quality-threshold 0.3 \
  --cluster-n-passes 3 \
  -v

echo
echo "=== Step 3: Video Enhancement ==="
"${PYTHON_BIN}" "${ROOT}/process_parallel.py" \
  --task video_enhance \
  --scenes-folder "${SCENES_DIR}" \
  --enhance-scale 2.0 \
  --enhance-denoise 1.0 \
  --enhance-crf 16 \
  --video-codec auto \
  -v

echo
echo "=== Step 4: Audio Enhancement ==="
"${PYTHON_BIN}" "${ROOT}/process_parallel.py" \
  --task audio_enhance \
  --scenes-folder "${SCENES_DIR}" \
  --audio-denoise -25 \
  --audio-bitrate 192k \
  -v

echo
echo "=== Step 5: Subtitles ==="
"${PYTHON_BIN}" "${ROOT}/process_parallel.py" \
  --task subtitles \
  --scenes-folder "${SCENES_DIR}" \
  -v

echo
echo "=== Pipeline Complete! ==="
echo "Results in: ${OUTPUT_DIR}"
echo "Face clusters: ${OUTPUT_DIR}/faces_export"
echo "Enhanced videos: ${SCENES_DIR} (look for *_enhanced.mp4)"
echo "Subtitles: ${SCENES_DIR} (look for *.srt files)"
