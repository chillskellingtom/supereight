"""
Test script to reproduce the TypeError issue with Whisper on files without audio.
"""
import sys
from pathlib import Path
import traceback

def test_whisper_on_video_without_audio():
    """Test what happens when we try to transcribe a video file with no audio stream."""
    try:
        import whisper
        print("✓ Whisper imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import whisper: {e}")
        sys.exit(1)

    # Load a tiny model for testing
    print("\nLoading whisper model...")
    try:
        model = whisper.load_model("tiny", device="cpu")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Try to transcribe a file that exists but has no audio
    test_file = Path(r"C:\Users\latch\connor_family_movies_processed\scenes\tape 1\tape 1-Scene-001.rife.1.realesrgan.mkv")

    if not test_file.exists():
        print(f"\n⚠️  Test file not found: {test_file}")
        print("Using a non-existent file instead to test error handling")
        test_file = Path("nonexistent.mp4")

    print(f"\nAttempting to transcribe: {test_file}")
    print("(This should fail because the file has no audio stream)\n")

    try:
        result = model.transcribe(str(test_file))
        print(f"✓ Transcription succeeded (unexpected): {result}")
    except RuntimeError as e:
        print(f"✓ Caught RuntimeError (expected):")
        print(f"  {e}")
    except TypeError as e:
        print(f"✗ Caught TypeError (unexpected):")
        print(f"  {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"✗ Caught unexpected exception type: {type(e).__name__}")
        print(f"  {e}")
        traceback.print_exc()

        # Check if the exception has a .shape attribute (this would be the bug)
        if hasattr(e, 'shape'):
            print(f"\n⚠️  Exception object has .shape attribute: {e.shape}")
        else:
            print(f"\n  Exception does NOT have .shape attribute")

if __name__ == "__main__":
    test_whisper_on_video_without_audio()
