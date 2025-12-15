"""
Test that the environment is correctly configured for the smoke test.
"""
import subprocess
import sys


def test_current_python_has_whisper():
    """The current Python interpreter should have whisper installed"""
    try:
        import whisper
        assert True, "Whisper is installed"
    except ImportError:
        assert False, "Current Python does not have whisper installed"


def test_smoke_script_uses_correct_python():
    """
    The smoke script should explicitly use the Python that has whisper.

    This test fails if the script relies on PATH, which may pick the wrong Python.
    """
    # Read the smoke script
    with open("scripts/smoke_subtitles_tape1_1to20.ps1", "r") as f:
        script_content = f.read()

    # Check if it uses absolute Python path or sys.executable
    uses_absolute_path = "AppData\\Local\\Programs\\Python\\Python311\\python.exe" in script_content
    uses_sys_executable = "sys.executable" in script_content

    # For now, this will fail - we'll fix it by updating the script
    assert uses_absolute_path or uses_sys_executable, \
        "Smoke script should use absolute Python path to avoid PATH issues"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
