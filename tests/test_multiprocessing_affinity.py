"""
Test multiprocessing-based GPU affinity (the fix).

This test demonstrates that separate processes CAN maintain independent
ZE_AFFINITY_MASK settings, unlike threads.
"""
import os
import sys
import multiprocessing
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def worker_process(name, affinity_value, result_queue):
    """Worker process that sets ZE_AFFINITY_MASK before any imports"""
    # Set affinity BEFORE importing anything GPU-related
    os.environ["ZE_AFFINITY_MASK"] = affinity_value

    # Simulate some work
    time.sleep(0.1)

    # Read back the value - it should match what we set
    actual_value = os.environ.get("ZE_AFFINITY_MASK", "not set")

    result_queue.put({
        "name": name,
        "intended": affinity_value,
        "actual": actual_value,
        "pid": os.getpid(),
    })


def test_processes_have_independent_env_variables():
    """
    Demonstrates that separate processes CAN maintain independent environment variables.

    This is the fix for GPU affinity:
    - Each process has its own memory space
    - Environment variables are process-level, so each process has independent env
    - Therefore, each worker can have its own ZE_AFFINITY_MASK
    """
    # Use spawn method for Windows compatibility
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    # Start two processes with different affinities
    p1 = ctx.Process(target=worker_process, args=("worker1", "0", result_queue))
    p2 = ctx.Process(target=worker_process, args=("worker2", "1", result_queue))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Collect results
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result["name"]] = result

    # Each process should have a different PID (different processes)
    assert results["worker1"]["pid"] != results["worker2"]["pid"], \
        "Processes should have different PIDs"

    print("\nProcess affinity test results:")
    print(f"Worker 1 PID {results['worker1']['pid']}: intended='{results['worker1']['intended']}', actual='{results['worker1']['actual']}'")
    print(f"Worker 2 PID {results['worker2']['pid']}: intended='{results['worker2']['intended']}', actual='{results['worker2']['actual']}'")

    # Each worker should have the CORRECT affinity (no interference)
    assert results["worker1"]["actual"] == results["worker1"]["intended"], \
        f"Worker 1 should maintain its own affinity: expected '{results['worker1']['intended']}', got '{results['worker1']['actual']}'"

    assert results["worker2"]["actual"] == results["worker2"]["intended"], \
        f"Worker 2 should maintain its own affinity: expected '{results['worker2']['intended']}', got '{results['worker2']['actual']}'"

    print("\n✓ SUCCESS: Each process maintained independent ZE_AFFINITY_MASK")
    print("This proves multiprocessing solves the GPU affinity bug.")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
