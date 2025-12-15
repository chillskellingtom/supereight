# Connor Family Movies - CODEX.md

Guidance for Codex when editing this repo.

## Operating style
- Default to small, well-scoped patches; keep comments short and functional.
- Prefer `rg` for search; use PowerShell runners for scripts; keep files ASCII.
- Avoid touching video/model assets unless asked.

## Environment
- Work from Windows PowerShell using conda env `ipex-llm-xpu` (Python 3.11+ with Intel XPU/IPEX/torch/whisper).
- GPU topology: dual Intel Arc A770; respect `ZE_AFFINITY_MASK` pinning and VRAM guardrails in `process_parallel.py`.

## Where to edit first
- Pipeline logic: `process_parallel.py`, `worker_bootstrap.py`.
- Automation: `scripts/run_all.ps1`, smoke tests in `scripts/smoke_subtitles_tape1_fast.ps1`.
- Tests: `tests/test_smoke_subtitles_fast.py`, `tests/test_gpu_affinity.py`, `tests/test_progress_monitoring.py`.

## Validation
- `pwsh -File scripts/smoke_subtitles_tape1_fast.ps1`
- `python -m pytest tests/test_smoke_subtitles_fast.py -v -s`
- For quick sanity while coding: `python process_parallel.py --task subtitles --limit 2 -v`

## Non-negotiables
- Preserve log markers `[worker-start]`, `[model-load]`, `[progress]` and affinity verification output.
- Keep VRAM checks and XPU-first execution paths; avoid CPU fallbacks unless asked.
- Respect `.gitignore` for media/log artifacts.
