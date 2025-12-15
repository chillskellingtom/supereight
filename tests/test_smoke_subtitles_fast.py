import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def _have_pwsh() -> bool:
    return shutil.which("pwsh") is not None or shutil.which("powershell") is not None


@pytest.mark.smoke
def test_fast_smoke_subtitles_two_scenes():
    if not _have_pwsh():
        pytest.skip("pwsh/powershell not found on PATH")

    repo_root = Path(__file__).parent.parent
    script = repo_root / "scripts" / "smoke_subtitles_tape1_fast.ps1"
    assert script.exists(), f"Missing smoke script: {script}"

    scenes_root = os.environ.get("SCENES_ROOT", r"C:\Users\latch\connor_family_movies_processed\scenes")
    if not Path(scenes_root).exists():
        pytest.skip(f"Scenes root not found: {scenes_root}")

    # Prefer pwsh (PowerShell 7) if present
    runner = "pwsh" if shutil.which("pwsh") else "powershell"

    # Target: <60s; give margin in CI while still enforcing "fast"
    # If this flakes early, bump to 120 and tighten once scene selection is confirmed.
    timeout_s = 120

    result = subprocess.run(
        [runner, "-NoProfile", "-File", str(script)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")

    assert result.returncode == 0, f"Fast smoke failed: rc={result.returncode}\n\n{combined}"

    # Invariant: total scenes should be 2 (from process_parallel logging)
    # Example: "Total scenes: 2 | workers: [...]"
    assert re.search(r"Total scenes:\s*2\b", combined), f"Expected Total scenes: 2\n\n{combined}"

    # Invariant: two workers start (two worker-start lines, two PIDs)
    worker_starts = re.findall(r"\[worker-start\]\s+PID=(\d+).*ZE_AFFINITY_MASK=([0-9.]+)", combined)
    assert len(worker_starts) == 2, f"Expected 2 worker-start lines, got {len(worker_starts)}\n\n{combined}"

    pids = {pid for pid, _mask in worker_starts}
    masks = {mask for _pid, mask in worker_starts}
    assert len(pids) == 2, f"Expected 2 distinct worker PIDs, got {pids}\n\n{combined}"
    assert len(masks) == 2, f"Expected 2 distinct ZE_AFFINITY_MASK values, got {masks}\n\n{combined}"
    assert (("0.0" in masks and "0.1" in masks) or ("0" in masks and "1" in masks)), \
        f"Expected masks 0/1 (or 0.0/1.0), got {masks}\n\n{combined}"

    # Invariant: model-load exactly once per worker (2 total, 2 distinct pids)
    model_loads = re.findall(r"\[model-load\]\s+PID=(\d+)", combined)
    assert len(model_loads) == 2, f"Expected exactly 2 model-load events, got {len(model_loads)}\n\n{combined}"
    assert len(set(model_loads)) == 2, f"Expected model-load from 2 distinct PIDs, got {set(model_loads)}\n\n{combined}"

    # Invariant: progress messages appear and reach (2/2)
    assert "[progress]" in combined, f"Expected at least one [progress] line\n\n{combined}"
    assert re.search(r"\(2/2\)", combined), f"Expected final progress to reach (2/2)\n\n{combined}"
