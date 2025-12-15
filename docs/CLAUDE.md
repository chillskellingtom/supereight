# Connor Family Movies - CLAUDE.md

This file provides guidance to Claude Code when working with this video processing pipeline.

## Project Overview

This project processes vintage family video footage using Intel Arc A770 GPUs for AI-enhanced video upscaling, frame interpolation, and audio transcription.

## Required Environment

**CRITICAL**: This project requires the `ipex-llm-xpu` conda environment.

- **Environment name**: `ipex-llm-xpu`
- **Python version**: 3.11+ (check with `conda list` in the environment)
- **Required packages**:
  - `intel-extension-for-pytorch` (IPEX)
  - `torch` with Intel XPU support
  - `openai-whisper` (for audio transcription)
  - `opencv-python` (for video processing)
  - Other dependencies in `requirements.txt` (if present)

### Activating the Environment

All Python scripts in this project **MUST** run in the `ipex-llm-xpu` environment.

**PowerShell scripts** (in `scripts/`) automatically activate the environment:
```powershell
conda run -n ipex-llm-xpu python process_parallel.py ...
```

**Manual activation** (if needed):
```bash
conda activate ipex-llm-xpu
```

**Verification**:
```bash
python -c "import intel_extension_for_pytorch as ipex; print(f'IPEX available: {ipex.xpu.is_available()}')"
```

## Hardware Configuration

- **GPUs**: 2x Intel Arc A770 Graphics (16GB VRAM each)
- **Backend**: Intel XPU (via IPEX/oneAPI)
- **GPU Affinity**: Each worker process is pinned to a specific GPU using `ZE_AFFINITY_MASK`
  - Worker 0 → GPU 0 (`ZE_AFFINITY_MASK=0`)
  - Worker 1 → GPU 1 (`ZE_AFFINITY_MASK=1`)

## Main Processing Scripts

- **`process_parallel.py`**: Main video processing pipeline (subtitles, enhancement, etc.)
  - Supports multi-GPU parallel processing
  - Automatic VRAM checking and model selection
  - Tasks: `subtitles`, `enhance`, `denoise`, etc.

- **Smoke tests** (in `scripts/`):
  - `smoke_subtitles_tape1_1to20.ps1`: Quick validation test (20 scenes)
  - `smoke_subtitles_tape1_fast.ps1`: Faster test variant

## Common Environment Variables

Set in PowerShell scripts before running:
- `PYTHONUNBUFFERED=1`: Real-time log output
- `LOG_LEVEL=INFO`: Logging verbosity
- `SYCL_PI_TRACE=0`: Disable verbose Level Zero API tracing
- `ZE_ENABLE_TRACING=0`: Disable GPU tracing
- `_VERIFY_GPU_AFFINITY=1`: Enable GPU affinity verification logs

## Known Issues & Fixes

### VRAM Detection (FIXED 2025-12-14)

**Issue**: VRAM check incorrectly reported 0 GB free, causing model downgrade to "tiny".

**Root cause**: Used `mem_reserved - mem_allocated` instead of `mem_total - mem_allocated`.
When GPU hasn't allocated memory yet, both values are 0.

**Fix**: Updated `check_vram_available()` in `process_parallel.py:518` to use:
```python
mem_free = mem_total - mem_allocated  # Correct calculation
```

## Testing & Validation

Run smoke tests to validate GPU detection and processing:

```powershell
.\scripts\smoke_subtitles_tape1_1to20.ps1
```

Expected output should show:
- ✅ Both GPUs detected via WMI
- ✅ GPU affinity verification passed
- ✅ Models loaded on `xpu:0` and `xpu:1` (not CPU)
- ✅ VRAM check shows >10 GB free (not 0 GB)
- ✅ Model selection: "medium" (not "tiny")

## Directory Structure

- `/scripts/`: PowerShell automation scripts
- `/tests/`: Unit and integration tests
- `/process_parallel.py`: Main processing pipeline
- Video input: `C:\Users\latch\connor_family_movies_processed\scenes\`

## Important Notes

- **DO NOT** use CPU fallback unless explicitly requested
- **ALWAYS** verify the `ipex-llm-xpu` environment is active
- **GPU affinity is critical** - each worker must be pinned to its GPU
- **VRAM checks** prevent out-of-memory errors during model loading
