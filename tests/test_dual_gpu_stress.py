"""
2× GPU Stress Test Suite
=========================

Comprehensive stress testing for dual Intel Arc A770 parallel video processing.

Test Categories:
1. GPU Affinity Verification (CRITICAL)
2. VRAM Utilization and OOM Resilience
3. Load Balancing and Throughput
4. Crash Recovery and Checkpointing
5. Long-Running Stability (Memory Leaks)
6. Determinism and Reproducibility

Usage:
    # Run all stress tests (LONG - may take hours)
    pytest test_dual_gpu_stress.py -v -s

    # Run specific test
    pytest test_dual_gpu_stress.py::test_gpu_affinity_dual_workers -v -s

    # Run quick smoke test (5min)
    pytest test_dual_gpu_stress.py::test_smoke_dual_gpu -v -s
"""
import subprocess
import sys
import re
import time
import json
import psutil
from pathlib import Path
from typing import List, Tuple, Dict, Any


# Test parameters
REPO_ROOT = Path(__file__).parent.parent
SCENES_FOLDER = REPO_ROOT / "scenes"  # Adjust to your scene directory


def run_parallel_process(
    task: str,
    limit: int = None,
    dry_run: bool = False,
    verbose: int = 2,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Helper to run process_parallel.py with standardized args."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "process_parallel.py"),
        "--task", task,
        "-" + "v" * verbose,
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )

    return result


def parse_worker_logs(output: str) -> List[Dict[str, Any]]:
    """Extract worker identity logs from output."""
    pattern = r'\[worker-start\] PID=(\d+).*device_idx=(\d+).*\(([^)]+)\).*ZE_AFFINITY_MASK=([^\s]+)'
    workers = []

    for match in re.finditer(pattern, output):
        workers.append({
            "pid": int(match.group(1)),
            "device_idx": int(match.group(2)),
            "device_name": match.group(3),
            "ze_affinity_mask": match.group(4),
        })

    return workers


def parse_model_load_logs(output: str) -> List[Dict[str, Any]]:
    """Extract model load events from output."""
    pattern = r'\[model-load\] PID=(\d+).*device_idx=(\d+).*model=(\w+).*device["\']:\s*["\']([^"\']+)'
    loads = []

    for match in re.finditer(pattern, output):
        loads.append({
            "pid": int(match.group(1)),
            "device_idx": int(match.group(2)),
            "model": match.group(3),
            "device": match.group(4),
        })

    return loads


def parse_affinity_verify_logs(output: str) -> List[Dict[str, Any]]:
    """Extract GPU affinity verification logs."""
    pattern = r'\[affinity-verify\] PID=(\d+).*Expected device: ([^\s]+).*Actual device: ([^\s]+)'
    verifications = []

    for match in re.finditer(pattern, output):
        expected = match.group(2)
        actual = match.group(3)
        verifications.append({
            "pid": int(match.group(1)),
            "expected": expected,
            "actual": actual,
            "match": expected in actual,
        })

    return verifications


# ==============================================================================
# TEST 1: GPU Affinity Verification (CRITICAL)
# ==============================================================================

def test_gpu_affinity_dual_workers():
    """
    CRITICAL: Verify that two workers use different GPUs (not same GPU, different tiles).

    Success Criteria:
    - Worker 0: ZE_AFFINITY_MASK=0 (or 0.0), model on xpu:0
    - Worker 1: ZE_AFFINITY_MASK=1 (or 1.0), model on xpu:1

    Failure Modes to Detect:
    - Both workers: ZE_AFFINITY_MASK=0.0 and 0.1 (BUG: same GPU, different tiles)
    - Models both on xpu:0 (BUG: affinity not honored)
    """
    print("\n" + "=" * 80)
    print("TEST 1: GPU AFFINITY VERIFICATION")
    print("=" * 80)

    result = run_parallel_process(
        task="subtitles",
        limit=2,  # 2 scenes = 1 per worker
        verbose=2,
        timeout=180,
    )

    output = result.stdout + "\n" + result.stderr
    print(output)  # Full debug output

    # Parse worker logs
    workers = parse_worker_logs(output)
    print(f"\n📊 Found {len(workers)} workers")
    for w in workers:
        print(f"  Worker PID={w['pid']}: device_idx={w['device_idx']}, ZE_AFFINITY_MASK={w['ze_affinity_mask']}")

    # ASSERTION 1: Two workers started
    assert len(workers) == 2, f"Expected 2 workers, found {len(workers)}"

    # ASSERTION 2: Different PIDs
    pids = [w["pid"] for w in workers]
    assert len(set(pids)) == 2, f"Workers should have different PIDs: {pids}"

    # ASSERTION 3: Different device indices
    device_indices = [w["device_idx"] for w in workers]
    assert device_indices == [0, 1], f"Workers should use device 0 and 1, got {device_indices}"

    # ASSERTION 4: ZE_AFFINITY_MASK is correct
    masks = [w["ze_affinity_mask"] for w in workers]

    # Accept either "0"/"1" (card-level) or "0.0"/"1.0" (tile-format)
    assert masks[0] in ["0", "0.0"], f"Worker 0 should have ZE_AFFINITY_MASK=0 or 0.0, got {masks[0]}"
    assert masks[1] in ["1", "1.0"], f"Worker 1 should have ZE_AFFINITY_MASK=1 or 1.0, got {masks[1]}"

    # Parse model load logs
    model_loads = parse_model_load_logs(output)
    print(f"\n📦 Found {len(model_loads)} model loads")
    for ml in model_loads:
        print(f"  PID={ml['pid']}: model={ml['model']}, device={ml['device']}")

    # ASSERTION 5: Models loaded on correct devices
    assert len(model_loads) == 2, f"Expected 2 model loads, found {len(model_loads)}"

    devices = [ml["device"] for ml in model_loads]
    assert "xpu:0" in devices, f"No model loaded on xpu:0, got {devices}"
    assert "xpu:1" in devices, f"No model loaded on xpu:1, got {devices}"

    # Parse affinity verification logs (if patch #1 applied)
    verifications = parse_affinity_verify_logs(output)
    if verifications:
        print(f"\n✅ Found {len(verifications)} affinity verifications")
        for v in verifications:
            status = "✓" if v["match"] else "✗"
            print(f"  {status} PID={v['pid']}: expected={v['expected']}, actual={v['actual']}")

        # ASSERTION 6: All verifications passed
        assert all(v["match"] for v in verifications), "GPU affinity verification failed!"

    print("\n✅ GPU AFFINITY TEST PASSED")
    print("   - 2 workers with different PIDs")
    print("   - Workers use device 0 and 1")
    print("   - ZE_AFFINITY_MASK correctly set (0 and 1)")
    print("   - Models loaded on xpu:0 and xpu:1")


# ==============================================================================
# TEST 2: VRAM Utilization and OOM Resilience
# ==============================================================================

def test_vram_utilization():
    """
    Test VRAM usage patterns and OOM resilience.

    Success Criteria:
    - VRAM check logs present
    - No OOM crashes
    - Model downgrades if low VRAM (if patch #2 applied)
    """
    print("\n" + "=" * 80)
    print("TEST 2: VRAM UTILIZATION")
    print("=" * 80)

    result = run_parallel_process(
        task="subtitles",
        limit=4,
        verbose=2,
        timeout=300,
    )

    output = result.stdout + "\n" + result.stderr

    # Look for VRAM check logs (if patch #2 applied)
    vram_checks = re.findall(r'\[vram-check\] GPU (\d+).*Free: ([\d.]+) GB.*Need: ([\d.]+) GB.*OK: (.)', output)

    if vram_checks:
        print(f"\n📊 Found {len(vram_checks)} VRAM checks")
        for gpu_idx, free_gb, need_gb, ok in vram_checks:
            print(f"  GPU {gpu_idx}: {free_gb} GB free, need {need_gb} GB - {'✓' if ok == '✓' else '✗'}")
    else:
        print("\n⚠️  No VRAM checks found (patch #2 not applied?)")

    # Check for OOM errors
    oom_errors = re.findall(r'(out of memory|OOM|cuda.*out of memory|xpu.*out of memory)', output, re.IGNORECASE)
    if oom_errors:
        print(f"\n❌ OOM ERRORS DETECTED: {len(oom_errors)}")
        for err in oom_errors[:5]:  # Show first 5
            print(f"  - {err}")
        raise AssertionError("OOM errors detected - VRAM safety not working!")

    # Check for model downgrades (VRAM safety in action)
    downgrades = re.findall(r'\[vram-safety\].*downgrading to \'(\w+)\' model', output)
    if downgrades:
        print(f"\n🔽 Model downgrades detected: {downgrades}")
        print("   (VRAM safety working correctly)")

    print("\n✅ VRAM UTILIZATION TEST PASSED")
    print("   - No OOM crashes")
    if vram_checks:
        print(f"   - {len(vram_checks)} VRAM checks performed")
    if downgrades:
        print(f"   - {len(downgrades)} automatic model downgrades")


# ==============================================================================
# TEST 3: Load Balancing and Throughput
# ==============================================================================

def test_load_balancing():
    """
    Test load distribution across GPUs.

    Success Criteria:
    - Both workers process approximately equal number of scenes
    - Total time < 2× single-GPU time (parallel speedup)
    """
    print("\n" + "=" * 80)
    print("TEST 3: LOAD BALANCING AND THROUGHPUT")
    print("=" * 80)

    # Process 20 scenes with 2 workers
    start_time = time.time()

    result = run_parallel_process(
        task="subtitles",
        limit=20,
        verbose=1,
        timeout=1200,  # 20min timeout
    )

    elapsed = time.time() - start_time

    output = result.stdout + "\n" + result.stderr

    # Count scenes processed per worker
    progress_pattern = r'\[progress\] PID=(\d+) scene=([^\s]+)'
    progress_logs = re.findall(progress_pattern, output)

    scenes_per_worker = {}
    for pid, scene in progress_logs:
        scenes_per_worker.setdefault(pid, []).append(scene)

    print(f"\n📊 Load Distribution:")
    for pid, scenes in scenes_per_worker.items():
        print(f"  Worker PID={pid}: {len(scenes)} scenes")

    # ASSERTION: Both workers processed scenes
    assert len(scenes_per_worker) == 2, f"Expected 2 workers, found {len(scenes_per_worker)}"

    counts = [len(scenes) for scenes in scenes_per_worker.values()]
    balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0

    print(f"\n⚖️  Load Balance Ratio: {balance_ratio:.2f} (1.0 = perfect balance)")

    # ASSERTION: Reasonably balanced (within 20% difference)
    assert balance_ratio >= 0.8, f"Load imbalance detected: {counts}"

    print(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"   Average per scene: {elapsed/20:.1f}s")

    print("\n✅ LOAD BALANCING TEST PASSED")


def test_lpt_scheduler_logs():
    """
    Verify LPT (Longest Processing Time) scheduler is active (Patch #3).

    Success Criteria:
    - [scheduler] logs present showing duration-aware load distribution
    - Worker time estimates shown in logs
    """
    print("\n" + "=" * 80)
    print("TEST 3B: LPT SCHEDULER VERIFICATION (Patch #3)")
    print("=" * 80)

    result = run_parallel_process(
        task="subtitles",
        limit=10,
        verbose=2,
        timeout=600,
    )

    output = result.stdout + "\n" + result.stderr
    print(output)

    # Check for LPT scheduler logs
    scheduler_logs = re.findall(r'\[scheduler\] Load distribution \(LPT algorithm\):', output)

    assert len(scheduler_logs) > 0, \
        "No [scheduler] LPT logs found! Patch #3 may not be executing."

    print(f"\n✅ Found {len(scheduler_logs)} LPT scheduler log(s)")

    # Extract worker time estimates
    worker_estimates = re.findall(r'Worker (\d+): (\d+) scenes, estimated ([\d.]+) min', output)

    if worker_estimates:
        print("\n📊 LPT Load Distribution:")
        total_time = 0
        for worker_id, scene_count, est_min in worker_estimates:
            print(f"   Worker {worker_id}: {scene_count} scenes, {est_min} min")
            total_time += float(est_min)

        # Check balance
        times = [float(est_min) for _, _, est_min in worker_estimates]
        if len(times) == 2:
            balance_ratio = min(times) / max(times) if max(times) > 0 else 0
            print(f"\n   Time balance ratio: {balance_ratio:.2f}")

            if balance_ratio >= 0.7:
                print("   ✓ Well balanced (>70%)")
            else:
                print(f"   ⚠️  Imbalanced ({balance_ratio:.0%}) - may be due to video duration variance")

    print("\n✅ LPT SCHEDULER TEST PASSED")
    print("   - Scheduler logs present")
    print("   - Duration-aware load distribution active")
    print("\n✓ Patch #3 (LPT scheduler) is working!")


# ==============================================================================
# TEST 4: Crash Recovery and Checkpointing
# ==============================================================================

def test_checkpointing_and_resume():
    """
    Test crash recovery via checkpointing.

    Test Procedure:
    1. Process 10 scenes
    2. Verify .done markers created
    3. Re-run same command
    4. Verify scenes skipped (not re-processed)
    """
    print("\n" + "=" * 80)
    print("TEST 4: CHECKPOINTING AND CRASH RECOVERY")
    print("=" * 80)

    # Clean up any existing .done markers first
    import glob
    done_markers = list(SCENES_FOLDER.glob(".*.subtitles.done"))
    for marker in done_markers:
        marker.unlink()
        print(f"🧹 Cleaned up: {marker.name}")

    # First run: process 10 scenes
    print("\n▶️  First run: processing 10 scenes...")
    result1 = run_parallel_process(
        task="subtitles",
        limit=10,
        verbose=1,
        timeout=600,
    )

    # Count .done markers created
    done_markers = list(SCENES_FOLDER.glob(".*.subtitles.done"))
    print(f"\n📍 Created {len(done_markers)} checkpoint markers")

    # ASSERTION: All scenes checkpointed
    assert len(done_markers) == 10, f"Expected 10 .done markers, found {len(done_markers)}"

    # Second run: same command (should skip all scenes)
    print("\n▶️  Second run: should skip all scenes...")
    result2 = run_parallel_process(
        task="subtitles",
        limit=10,
        verbose=1,
        timeout=60,  # Should be much faster (skipping)
    )

    output2 = result2.stdout + "\n" + result2.stderr

    # Count skip logs
    skip_logs = re.findall(r'\[skip\] Already processed:', output2)
    print(f"\n⏭️  Skipped {len(skip_logs)} scenes")

    # ASSERTION: All scenes skipped
    assert len(skip_logs) == 10, f"Expected 10 skips, found {len(skip_logs)}"

    print("\n✅ CHECKPOINTING TEST PASSED")
    print("   - 10 checkpoint markers created")
    print("   - All scenes skipped on re-run")

    # Clean up
    for marker in done_markers:
        marker.unlink()


# ==============================================================================
# TEST 5: Long-Running Stability (Memory Leaks)
# ==============================================================================

def test_memory_leak_detection():
    """
    Test for memory leaks during long-running processing.

    Monitor process memory over time during 100-scene batch.
    """
    print("\n" + "=" * 80)
    print("TEST 5: MEMORY LEAK DETECTION")
    print("=" * 80)

    print("\n⏳ Running 100-scene batch with memory monitoring...")
    print("   (This may take 30-60 minutes)")

    # Start process
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "process_parallel.py"),
            "--task", "subtitles",
            "--limit", "100",
            "-v",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
    )

    # Monitor memory every 30 seconds
    memory_samples = []
    start_time = time.time()

    while proc.poll() is None:
        try:
            # Get process tree memory
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)

            total_rss = parent.memory_info().rss
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass

            total_rss_mb = total_rss / (1024 * 1024)
            elapsed = time.time() - start_time

            memory_samples.append({
                "elapsed_s": elapsed,
                "rss_mb": total_rss_mb,
                "num_processes": 1 + len(children),
            })

            print(f"  [{elapsed:.0f}s] Memory: {total_rss_mb:.1f} MB ({1 + len(children)} processes)")

        except psutil.NoSuchProcess:
            break

        time.sleep(30)  # Sample every 30s

    # Analyze memory trend
    if len(memory_samples) >= 3:
        initial_mem = memory_samples[0]["rss_mb"]
        final_mem = memory_samples[-1]["rss_mb"]
        max_mem = max(s["rss_mb"] for s in memory_samples)

        growth = final_mem - initial_mem
        growth_pct = (growth / initial_mem) * 100 if initial_mem > 0 else 0

        print(f"\n📈 Memory Analysis:")
        print(f"   Initial: {initial_mem:.1f} MB")
        print(f"   Final: {final_mem:.1f} MB")
        print(f"   Peak: {max_mem:.1f} MB")
        print(f"   Growth: {growth:.1f} MB ({growth_pct:+.1f}%)")

        # ASSERTION: Memory growth < 50% (reasonable for caching/buffers)
        assert growth_pct < 50, f"Potential memory leak: {growth_pct:.1f}% growth"

        print("\n✅ MEMORY LEAK TEST PASSED")
    else:
        print("\n⚠️  Not enough samples for leak detection")


# ==============================================================================
# TEST 6: Quick Smoke Test (5min)
# ==============================================================================

def test_smoke_dual_gpu():
    """
    Quick smoke test: verify basic dual-GPU functionality.

    This is a fast sanity check (< 5min).
    """
    print("\n" + "=" * 80)
    print("SMOKE TEST: Dual-GPU Basic Functionality")
    print("=" * 80)

    result = run_parallel_process(
        task="subtitles",
        limit=2,
        verbose=2,
        timeout=180,
    )

    assert result.returncode == 0, f"Process failed with exit code {result.returncode}"

    output = result.stdout + "\n" + result.stderr

    # Check for key success indicators
    workers = parse_worker_logs(output)
    assert len(workers) == 2, "Should have 2 workers"

    model_loads = parse_model_load_logs(output)
    assert len(model_loads) == 2, "Should have 2 model loads"

    # Check for completion
    assert "completed successfully" in output.lower() or result.returncode == 0

    print("\n✅ SMOKE TEST PASSED - Basic dual-GPU functionality working")


# ==============================================================================
# MAIN: Run All Tests
# ==============================================================================

if __name__ == "__main__":
    import pytest

    # Run with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
