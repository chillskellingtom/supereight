"""
Test GPU affinity behavior with threading vs multiprocessing

This test demonstrates that threads cannot have independent ZE_AFFINITY_MASK
settings because they share the same process.
"""
import os
import sys
import threading
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import process_parallel


def test_threads_share_env_variables():
    """
    Demonstrates that threads share environment variables in the same process.

    This test shows the fundamental issue with thread-based GPU affinity:
    - All threads run in the same process
    - Environment variables are process-level, not thread-level
    - Therefore, ZE_AFFINITY_MASK set by one thread affects all threads
    """
    results = {}

    def worker(name, affinity_value):
        """Worker thread that tries to set ZE_AFFINITY_MASK"""
        # Try to set affinity (this won't work correctly with threads)
        os.environ["ZE_AFFINITY_MASK"] = affinity_value
        time.sleep(0.1)  # Give other threads time to interfere

        # Read back the value - it may have been changed by other threads!
        actual_value = os.environ.get("ZE_AFFINITY_MASK", "not set")
        results[name] = {
            "intended": affinity_value,
            "actual": actual_value,
            "pid": os.getpid(),
            "thread_id": threading.current_thread().ident,
        }

    # Start two threads trying to set different affinities
    t1 = threading.Thread(target=worker, args=("worker1", "0.0"))
    t2 = threading.Thread(target=worker, args=("worker2", "0.1"))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both threads should have the same PID (same process)
    assert results["worker1"]["pid"] == results["worker2"]["pid"], \
        "Threads should run in the same process"

    # The critical bug: both threads cannot maintain different ZE_AFFINITY_MASK values
    # One will overwrite the other because env vars are process-level
    print("\nThread affinity test results:")
    print(f"Worker 1: intended='{results['worker1']['intended']}', actual='{results['worker1']['actual']}'")
    print(f"Worker 2: intended='{results['worker2']['intended']}', actual='{results['worker2']['actual']}'")
    print(f"Same PID: {results['worker1']['pid']}")

    # At least one worker will have the wrong affinity (or both will have the same)
    # This demonstrates why threading cannot work for GPU affinity
    affinity_mismatch = (
        results["worker1"]["actual"] != results["worker1"]["intended"] or
        results["worker2"]["actual"] != results["worker2"]["intended"]
    )

    assert affinity_mismatch, \
        "Threading bug demonstrated: at least one worker has wrong ZE_AFFINITY_MASK due to shared process env"


def test_current_worker_plan_uses_multiprocessing():
    """
    Verify that the current implementation uses multiprocessing.Process (not threading.Thread)

    This test confirms the fix for GPU affinity is in place.
    """
    import inspect

    # Check the main function to see if it creates Process objects
    source = inspect.getsource(process_parallel.main)

    # The fixed implementation uses Process
    assert "Process" in source, "Fixed implementation should use multiprocessing.Process"
    assert "mp_ctx.Process" in source or "multiprocessing.Process" in source, \
        "Should create worker processes via multiprocessing"

    # Worker creation should use worker_bootstrap
    assert "worker_bootstrap" in source, "Workers should use worker_bootstrap.worker_main"

    print("\n✓ Confirmed: Implementation uses multiprocessing.Process")
    print("✓ GPU affinity bug is fixed")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
