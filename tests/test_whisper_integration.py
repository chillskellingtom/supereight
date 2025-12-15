"""
Test whisper installation and smoke test invariants.

This test ensures:
1. Whisper is available
2. Smoke test can run
3. Logs show required invariants (PIDs, affinity, model loading)
"""
import subprocess
import sys
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_whisper_is_installed():
    """Verify openai-whisper is installed and importable"""
    try:
        import whisper
        assert hasattr(whisper, 'load_model'), "whisper module missing load_model function"
        print(f"✓ Whisper installed: {whisper.__file__}")
    except ImportError as e:
        # This test should fail first, prompting installation
        assert False, f"openai-whisper not installed: {e}\nRun: pip install openai-whisper"


def test_smoke_script_exists():
    """Verify smoke test script exists"""
    smoke_script = Path(__file__).parent.parent / "scripts" / "smoke_subtitles_tape1_1to20.ps1"
    assert smoke_script.exists(), f"Smoke script not found: {smoke_script}"


def test_smoke_run_completes_successfully():
    """
    Run smoke test and validate it completes with exit 0.

    This test will fail until whisper is installed and the pipeline works correctly.
    """
    smoke_script = Path(__file__).parent.parent / "scripts" / "smoke_subtitles_tape1_1to20.ps1"

    # Run smoke test with timeout (measured ~390s for 20 scenes, allow 600s for safety)
    result = subprocess.run(
        ["powershell", "-File", str(smoke_script)],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout (measured: ~390s + margin)
    )

    # Print output for debugging
    print("\n=== SMOKE TEST STDOUT ===")
    print(result.stdout)
    print("\n=== SMOKE TEST STDERR ===")
    print(result.stderr)

    assert result.returncode == 0, f"Smoke test failed with exit code {result.returncode}"


def test_smoke_logs_show_dual_gpu_parallelism():
    """
    Validate smoke test logs prove dual-GPU parallelism.

    Required invariants:
    1. Two distinct worker PIDs
    2. ZE_AFFINITY_MASK=0/1 (or 0.0/1.0) in worker-start logs
    3. Model loads once per worker (not per scene)
    """
    smoke_script = Path(__file__).parent.parent / "scripts" / "smoke_subtitles_tape1_1to20.ps1"

    # Run smoke test (measured ~390s for 20 scenes, allow 600s for safety)
    result = subprocess.run(
        ["powershell", "-File", str(smoke_script)],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout (measured: ~390s + margin)
    )

    combined_output = result.stdout + "\n" + result.stderr

    # Extract worker-start logs
    worker_start_pattern = r'\[worker-start\] PID=(\d+).*device_idx=(\d+).*ZE_AFFINITY_MASK=([\d.]+|not set)'
    worker_starts = re.findall(worker_start_pattern, combined_output)

    print(f"\n=== Found {len(worker_starts)} worker-start logs ===")
    for pid, dev_idx, ze_mask in worker_starts:
        print(f"  PID={pid}, device_idx={dev_idx}, ZE_AFFINITY_MASK={ze_mask}")

    # Invariant 1: At least 2 workers started (for 2 GPUs)
    assert len(worker_starts) >= 2, f"Expected at least 2 workers, found {len(worker_starts)}"

    # Invariant 2: Workers have distinct PIDs
    pids = [pid for pid, _, _ in worker_starts]
    unique_pids = set(pids)
    assert len(unique_pids) >= 2, f"Expected at least 2 distinct PIDs, found {unique_pids}"
    print(f"✓ Invariant: {len(unique_pids)} distinct worker PIDs")

    # Invariant 3/4: At least one worker per GPU (accept card-level 0/1 or tile-format 0.0/0.1)
    ze_masks = [ze_mask for _, _, ze_mask in worker_starts]
    has_gpu0 = ("0.0" in ze_masks) or ("0" in ze_masks)
    has_gpu1 = ("1.0" in ze_masks) or ("1" in ze_masks)
    assert has_gpu0, f"Expected ZE_AFFINITY_MASK for GPU0 (0 or 0.0), found masks: {ze_masks}"
    assert has_gpu1, f"Expected ZE_AFFINITY_MASK for GPU1 (1 or 1.0), found masks: {ze_masks}"
    print(f"✓ Invariant: Found ZE_AFFINITY_MASK values for both GPUs (masks: {ze_masks})")

    # Invariant 5: Model loading happens exactly once per worker
    # Look for [model-load] PID=... logs
    model_load_pattern = r'\[model-load\] PID=(\d+)'
    model_loads = re.findall(model_load_pattern, combined_output)
    model_load_pids = set(model_loads)
    print(f"✓ Found {len(model_loads)} model load events from {len(model_load_pids)} distinct PIDs")

    # We expect model to load once per worker, not per scene
    # With 2 workers and limit=20, we should see exactly 2 model loads
    assert len(model_loads) == 2, f"Expected 2 model loads (one per worker), found {len(model_loads)}"
    assert len(model_load_pids) == 2, f"Expected 2 distinct PIDs in model loads, found {len(model_load_pids)}"
    print(f"✓ Invariant: Model loads exactly once per worker (not per scene)")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
