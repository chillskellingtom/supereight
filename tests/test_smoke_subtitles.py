"""
Smoke tests for process_parallel.py

These tests ensure basic functionality works before deploying to full dataset.
"""
import pytest
import subprocess
import sys
from pathlib import Path

# Add parent directory to path so we can import process_parallel
sys.path.insert(0, str(Path(__file__).parent.parent))

import process_parallel


class TestLimit:
    """Test --limit flag functionality"""

    def test_limit_flag_accepted_by_parser(self):
        """Parser should accept --limit argument"""
        args = process_parallel.parse_args(["--limit", "5"])
        assert hasattr(args, "limit")
        assert args.limit == 5

    def test_limit_defaults_to_none(self):
        """Limit should default to None (no limit)"""
        args = process_parallel.parse_args([])
        assert hasattr(args, "limit")
        assert args.limit is None

    def test_gather_scenes_respects_limit(self):
        """gather_scenes should respect limit parameter"""
        # Use parent scenes folder (not tape 1 subfolder) - matches smoke script
        scenes_folder = Path(r"C:\Users\latch\connor_family_movies_processed\scenes")
        if not scenes_folder.exists():
            pytest.skip(f"Test scenes folder not found: {scenes_folder}")

        # Gather all scenes
        all_scenes = process_parallel.gather_scenes(scenes_folder)
        if len(all_scenes) < 5:
            pytest.skip(f"Not enough scenes for limit test (found {len(all_scenes)})")

        # Now test that gather_scenes respects a limit
        # We'll need to modify gather_scenes to accept a limit parameter
        limited_scenes = process_parallel.gather_scenes(scenes_folder, limit=5)
        assert len(limited_scenes) == 5
        assert limited_scenes == all_scenes[:5]


class TestSmokeSubtitles:
    """Smoke test for subtitle generation"""

    def test_process_parallel_runs_without_error(self):
        """process_parallel.py should run successfully with --limit 2"""
        # Use parent scenes folder (not tape 1 subfolder) - matches smoke script
        scenes_folder = Path(r"C:\Users\latch\connor_family_movies_processed\scenes")
        if not scenes_folder.exists():
            pytest.skip(f"Test scenes folder not found: {scenes_folder}")

        # Run with dry-run to test planning without actual processing
        result = subprocess.run(
            [
                sys.executable,
                "process_parallel.py",
                "--task", "subtitles",
                "--scenes-folder", str(scenes_folder),
                "--limit", "2",
                "--dry-run",
                "-v"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Total scenes:" in result.stdout or "Total scenes:" in result.stderr
