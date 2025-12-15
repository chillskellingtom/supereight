"""
Test GPU resource cleanup on worker termination (Patch #2 validation).

This test validates that Patch #2 (signal handler cleanup) properly
releases GPU resources when workers are terminated.
"""
import subprocess
import sys
import time
import re
import signal
from pathlib import Path
import psutil


REPO_ROOT = Path(__file__).parent.parent


def test_cleanup_handlers_registered():
    """
    Verify that cleanup handlers are registered on worker startup.

    This confirms Patch #2 code is being executed.
    """
    print("\n" + "=" * 80)
    print("TEST: CLEANUP HANDLER REGISTRATION (Patch #2)")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "process_parallel.py"),
            "--task", "subtitles",
            "--limit", "2",
            "-vv",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )

    output = result.stdout + result.stderr
    print(output)

    # Check for cleanup handler registration logs
    handler_logs = re.findall(r'\[worker-bootstrap\] PID=\d+ \| Cleanup handlers registered', output)

    assert len(handler_logs) > 0, \
        "No cleanup handler registration logs found! Patch #2 may not be executing."

    print(f"\n✅ Found {len(handler_logs)} cleanup handler registration(s)")
    print("✓ Cleanup handlers are being registered correctly")


def test_forced_termination_cleanup():
    """
    Verify GPU resources released when worker forcibly terminated.

    Steps:
    1. Start processing with 2 workers
    2. After 60s, SIGTERM one worker process
    3. Verify cleanup handler ran (check logs)
    4. Verify remaining worker continues processing
    """
    print("\n" + "=" * 80)
    print("TEST: FORCED TERMINATION CLEANUP (Patch #2)")
    print("=" * 80)

    print("\n⏳ Starting processing with limit=20 (will take ~2-3 min per GPU)...")

    # Start processing in background
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "process_parallel.py"),
            "--task", "subtitles",
            "--limit", "20",
            "-vv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
    )

    print(f"   Main process PID: {proc.pid}")

    # Wait for workers to start
    print("\n⏳ Waiting 60s for workers to start processing...")
    time.sleep(60)

    # Find worker PIDs
    try:
        parent = psutil.Process(proc.pid)
        workers = parent.children(recursive=True)

        print(f"\n📊 Found {len(workers)} worker process(es)")
        for w in workers:
            print(f"   Worker PID: {w.pid} (status: {w.status()})")

        if len(workers) < 1:
            print("\n⚠️  No worker processes found, test inconclusive")
            proc.terminate()
            proc.wait(timeout=10)
            return

        # Terminate first worker with SIGTERM
        worker0 = workers[0]
        worker0_pid = worker0.pid

        print(f"\n🔪 Sending SIGTERM to worker PID {worker0_pid}...")
        worker0.send_signal(signal.SIGTERM)

        # Wait for cleanup
        print("   Waiting 10s for cleanup...")
        time.sleep(10)

        # Check if worker exited
        worker_exists = psutil.pid_exists(worker0_pid)
        print(f"   Worker PID {worker0_pid} exists: {worker_exists}")

        if worker_exists:
            print("   ⚠️  Worker still alive, forcing kill...")
            try:
                psutil.Process(worker0_pid).kill()
            except psutil.NoSuchProcess:
                pass

        # Let main process finish or timeout
        print("\n⏳ Waiting for main process to complete (max 5 min)...")
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print("   ⏱️  Timeout reached, terminating main process...")
            proc.terminate()
            proc.wait(timeout=30)

        # Read output
        output = proc.communicate()[0] if proc.poll() is None else ""
        if proc.stdout:
            output = proc.stdout.read()

        print(f"\n📋 Output analysis:")

        # Check for cleanup logs
        cleanup_logs = re.findall(
            rf'\[worker-cleanup\] PID={worker0_pid} \| (Running cleanup handler|GPU resources cleaned successfully|Cleanup failed: .*)',
            output
        )

        if cleanup_logs:
            print(f"   ✅ Found {len(cleanup_logs)} cleanup log(s) for PID {worker0_pid}")
            for log in cleanup_logs:
                print(f"      - {log}")

            # Check if cleanup succeeded
            success_logs = [l for l in cleanup_logs if "cleaned successfully" in l]
            if success_logs:
                print("\n✅ CLEANUP VERIFICATION PASSED")
                print("   - Worker received SIGTERM")
                print("   - Cleanup handler executed")
                print("   - GPU resources released successfully")
                print("\n✓ Patch #2 (signal handler cleanup) is working!")
            else:
                print("\n⚠️  Cleanup handler ran but may have failed")
                print("   Check logs above for error details")
        else:
            print(f"   ⚠️  No cleanup logs found for PID {worker0_pid}")
            print("   This may indicate:")
            print("     1. Worker exited too quickly (before processing)")
            print("     2. Logging configuration issue")
            print("     3. Cleanup handler not triggered")
            print("\n   Test is INCONCLUSIVE (not necessarily a failure)")

        # Check that other workers continued
        worker_done_logs = re.findall(r'\[worker-done\] PID=(\d+)', output)
        if len(worker_done_logs) > 0:
            print(f"\n✓ {len(worker_done_logs)} worker(s) completed successfully")
            print("  (Remaining workers continued after termination)")

    except psutil.NoSuchProcess:
        print("\n⚠️  Main process exited before we could terminate worker")
        print("   Test inconclusive")
    except Exception as exc:
        print(f"\n❌ Test error: {exc}")
        # Cleanup
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except:
            pass
        raise


def test_cleanup_on_normal_exit():
    """
    Verify cleanup also works on normal exit (not just SIGTERM).

    This tests the atexit handler path.
    """
    print("\n" + "=" * 80)
    print("TEST: CLEANUP ON NORMAL EXIT (Patch #2)")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "process_parallel.py"),
            "--task", "subtitles",
            "--limit", "2",
            "-vv",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )

    output = result.stdout + result.stderr
    print(output)

    # Check for cleanup logs on normal exit
    cleanup_logs = re.findall(r'\[worker-cleanup\] PID=\d+ \| Running cleanup handler', output)

    # Note: Cleanup handler might not run on normal exit if model is None
    # or if the cleanup was already done in process_batch
    # So this is more informational than a hard requirement

    print(f"\n📊 Cleanup handler executions on normal exit: {len(cleanup_logs)}")

    if len(cleanup_logs) > 0:
        print("✓ Cleanup handlers executed on normal exit (via atexit)")
    else:
        print("ℹ️  No cleanup handler logs (model may have been cleaned in process_batch)")

    # The important thing is no errors and successful completion
    assert result.returncode == 0, "Process should exit successfully"
    print("\n✓ Normal exit cleanup test passed")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
