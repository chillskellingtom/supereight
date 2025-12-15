# Connor Family Movies

Fast GPU subtitle pipeline and supporting smoke tests.

Docs have moved to `docs/` (setup, GPU notes, reviews, stress plans).

## Fast GPU Smoke Test (Subtitles)

Run **only** from native Windows PowerShell (not WSL/Git Bash/VS Code Remote).

Requirements:
- Windows with Intel Arc GPUs
- Python 3.11 + pytest
- `pwsh` or `powershell.exe` on PATH

Commands:
```
pwsh -File scripts/smoke_subtitles_tape1_fast.ps1
python -m pytest tests/test_smoke_subtitles_fast.py -v -s
```

Expected markers:
- Two `[worker-start]` lines
- `ZE_AFFINITY_MASK=0.0` and `0.1`
- Two `[model-load]`
- Progress reaches `(2/2)`
- Exit code 0

Will **not** run from:
- WSL
- Git Bash
- Linux CI runners

## One-touch run
- `pwsh -File scripts/run_all.ps1` (sets oneAPI env, caches models, runs full pipeline)

## Parking Lot / Next Steps
- Optional: add advanced/alternate detectors (e.g., SAM-based or higher-precision face models) behind a feature flag with proper preprocessing/postprocessing.
