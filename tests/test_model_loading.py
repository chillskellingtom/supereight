"""
Test model loading behavior with multiprocessing workers.

This test ensures the critical performance invariant:
- Each worker loads the model EXACTLY ONCE (not per-scene)
"""
import subprocess
import sys
import re
from pathlib import Path


def test_model_loads_exactly_once_per_worker():
    """
    Prove that each worker process loads the model exactly once.

    This is a critical performance invariant. If the model reloads per scene,
    processing time would increase dramatically (model load is ~5-10 seconds).

    Required log contract:
    - [model-load] PID=... device_idx=... model=... backend=...
    - Exactly ONE per distinct worker PID
    """
    # Run with limit=2 (1 scene per worker with 2 GPUs)
    # This is faster for testing while still proving the invariant
    result = subprocess.run(
        [
            sys.executable,
            "process_parallel.py",
            "--task", "subtitles",
            "--limit", "2",
            "-v"
        ],
        capture_output=True,
        text=True,
        timeout=180,  # Generous timeout for model loading + transcription
        cwd=Path(__file__).parent.parent
    )

    combined_output = result.stdout + "\n" + result.stderr

    # Extract model-load events
    model_load_pattern = r'\[model-load\] PID=(\d+).*device_idx=(\d+).*model=(\w+).*backend=(\w+)'
    model_loads = re.findall(model_load_pattern, combined_output)

    print(f"\n=== Found {len(model_loads)} model-load events ===")
    for pid, dev_idx, model, backend in model_loads:
        print(f"  PID={pid}, device_idx={dev_idx}, model={model}, backend={backend}")

    # Also extract worker-start events to know how many workers we expect
    worker_start_pattern = r'\[worker-start\] PID=(\d+)'
    worker_pids = set(re.findall(worker_start_pattern, combined_output))

    print(f"\n=== Found {len(worker_pids)} distinct worker PIDs ===")
    for pid in sorted(worker_pids):
        print(f"  PID={pid}")

    # INVARIANT 1: We should have model-load events
    assert len(model_loads) > 0, \
        "No [model-load] events found. Add explicit logging to _load_model_for_device()"

    # INVARIANT 2: Exactly one model-load per worker PID
    load_pids = [pid for pid, _, _, _ in model_loads]
    unique_load_pids = set(load_pids)

    assert len(load_pids) == len(unique_load_pids), \
        f"Model loaded multiple times in same worker! PIDs: {load_pids}"

    # INVARIANT 3: Every worker that started should have loaded a model exactly once
    assert unique_load_pids == worker_pids, \
        f"Mismatch: worker PIDs {worker_pids} vs model-load PIDs {unique_load_pids}"

    print("\n✓ INVARIANT PROVEN: Each worker loaded model exactly once")

    # INVARIANT 4: With limit=2 and 2 workers, we expect exactly 2 model loads
    assert len(model_loads) == len(worker_pids), \
        f"Expected {len(worker_pids)} model loads (one per worker), found {len(model_loads)}"

    print(f"✓ PERFORMANCE INVARIANT: {len(model_loads)} model loads for {len(worker_pids)} workers processing 2 scenes")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
