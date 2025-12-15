# Connor Family Movies - CLAUDE.md

Fast guide for Claude Code when working on the Intel Arc subtitle/video pipeline.

## What this repo does
- Processes family tapes with GPU-based speech-to-text, scene detection, and VRT enhancement.
- Primary entry point is `process_parallel.py` with tasks such as `subtitles`, `enhance`, and monitoring in `worker_bootstrap.py`.

## Environment and platform
- Run from Windows PowerShell (not WSL or Git Bash).
- Use conda env `ipex-llm-xpu` (Python 3.11+, Intel XPU/IPEX/torch + whisper dependencies).
- Hardware target: dual Intel Arc A770 GPUs; respect `ZE_AFFINITY_MASK` pinning.

## Critical expectations
- Keep GPU affinity validation and VRAM safety checks intact.
- Preserve log markers consumed by tests (`[worker-start]`, `[model-load]`, `[progress]`).
- Prefer XPU paths over CPU fallback unless explicitly requested.
- Large media and model artifacts belong on disk, not in git.

## Quick commands
- `pwsh -File scripts/smoke_subtitles_tape1_fast.ps1`
- `python -m pytest tests/test_smoke_subtitles_fast.py -v -s`
- `python -m pytest tests/test_gpu_affinity.py -v -s`
- `python process_parallel.py --task subtitles --limit 2 -v`

## Pointers
- More background: `docs/CLAUDE.md`, `docs/INTEL_GPU_SETUP.md`, `docs/PATCHES_APPLIED.md`, `README.md`.
- Scripts: `scripts/*.ps1` set env vars and call the pipeline; adjust there before code changes when possible.
- Tests rely on deterministic output and mask handling; update fixtures/log expectations when behavior legitimately changes.
