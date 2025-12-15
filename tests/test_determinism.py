"""
Test deterministic output behavior (Patch #1 validation).

This test validates that Patch #1 (RNG seeding) produces identical
transcription results across multiple runs.
"""
import subprocess
import sys
import hashlib
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).parent.parent
SCENES_FOLDER = REPO_ROOT / r"C:\Users\latch\connor_family_movies_processed\scenes"


def get_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_deterministic_transcription_single_scene():
    """
    Run same scene 3 times, verify identical SRT outputs.

    This is the critical test for Patch #1 (RNG seeding).
    """
    print("\n" + "=" * 80)
    print("TEST: DETERMINISTIC TRANSCRIPTION (Patch #1)")
    print("=" * 80)

    # Find a test scene
    scene_files = list(SCENES_FOLDER.rglob("*.mp4"))
    if not scene_files:
        print("⚠️  No scene files found, skipping test")
        return

    test_scene = scene_files[0]
    print(f"\n📹 Test scene: {test_scene.name}")

    # Create temporary test directories
    test_dirs = []
    for run_idx in range(3):
        test_dir = REPO_ROOT / f"test_determinism_run{run_idx}"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True)
        test_dirs.append(test_dir)

    try:
        srt_hashes = []
        srt_contents = []

        for run_idx, test_dir in enumerate(test_dirs):
            print(f"\n▶️  Run {run_idx + 1}/3: Processing scene...")

            # Run processing
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "process_parallel.py"),
                    "--task", "subtitles",
                    "--scene-file", str(test_scene),
                    "--run-dir", str(test_dir),
                    "-v",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=REPO_ROOT,
            )

            assert result.returncode == 0, f"Run {run_idx + 1} failed: {result.stderr}"

            # Find generated SRT file
            srt_files = list(test_dir.rglob("*.srt"))
            assert len(srt_files) == 1, f"Expected 1 SRT file, found {len(srt_files)}"

            srt_file = srt_files[0]
            print(f"   Generated: {srt_file.name}")

            # Compute hash
            file_hash = get_file_hash(srt_file)
            srt_hashes.append(file_hash)
            print(f"   SHA256: {file_hash[:16]}...")

            # Store content for detailed comparison if needed
            with open(srt_file, 'r', encoding='utf-8') as f:
                srt_contents.append(f.read())

            # Check logs for seed confirmation
            output = result.stdout + result.stderr
            if "[seed]" in output:
                import re
                seed_logs = re.findall(r'\[seed\].*worker_seed=(\d+)', output)
                if seed_logs:
                    print(f"   ✓ Seed set: worker_seed={seed_logs[0]}")

        # CRITICAL ASSERTION: All hashes must be identical
        print(f"\n📊 Results:")
        print(f"   Run 1 hash: {srt_hashes[0][:16]}...")
        print(f"   Run 2 hash: {srt_hashes[1][:16]}...")
        print(f"   Run 3 hash: {srt_hashes[2][:16]}...")

        if srt_hashes[0] == srt_hashes[1] == srt_hashes[2]:
            print("\n✅ DETERMINISM VERIFIED: All 3 runs produced identical output")
            print("   Patch #1 (RNG seeding) is working correctly!")
        else:
            # Show differences for debugging
            print("\n❌ DETERMINISM FAILED: Outputs differ across runs")
            print("\nContent comparison:")
            if srt_contents[0] != srt_contents[1]:
                print("  Run 1 vs Run 2: DIFFERENT")
            if srt_contents[0] != srt_contents[2]:
                print("  Run 1 vs Run 3: DIFFERENT")
            if srt_contents[1] != srt_contents[2]:
                print("  Run 2 vs Run 3: DIFFERENT")

            raise AssertionError(
                "Transcription not deterministic! "
                f"Hashes: {srt_hashes[0][:8]} vs {srt_hashes[1][:8]} vs {srt_hashes[2][:8]}"
            )

    finally:
        # Cleanup test directories
        for test_dir in test_dirs:
            if test_dir.exists():
                shutil.rmtree(test_dir)


def test_seed_logs_present():
    """
    Verify that seed logging is present in worker output.

    This confirms Patch #1 code is being executed.
    """
    print("\n" + "=" * 80)
    print("TEST: SEED LOGGING VERIFICATION")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "process_parallel.py"),
            "--task", "subtitles",
            "--limit", "1",
            "-vv",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )

    output = result.stdout + result.stderr
    print(output)

    # Check for seed logs
    import re
    seed_logs = re.findall(r'\[seed\] PID=\d+ \| device_idx=\d+ \| Seeded RNGs with worker_seed=(\d+)', output)

    assert len(seed_logs) > 0, \
        "No [seed] logs found! Patch #1 may not be executing correctly."

    print(f"\n✅ Found {len(seed_logs)} seed log(s)")
    for idx, seed in enumerate(seed_logs):
        print(f"   Worker {idx}: seed={seed}")

    print("\n✓ Seed logging is working correctly")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
