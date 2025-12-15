"""
Test script to reproduce the TypeError issue with Whisper on Intel XPU backend.
"""
import sys
import os
from pathlib import Path
import traceback

def test_whisper_on_xpu():
    """Test whisper on Intel XPU backend (matching production)."""
    try:
        import whisper
        import torch
        print("✓ Whisper and torch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        sys.exit(1)

    # Check if XPU is available
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✓ Intel XPU is available")
        device = "xpu:0"
        os.environ["ZE_AFFINITY_MASK"] = "0"
    else:
        print("⚠️  Intel XPU not available, using CPU")
        device = "cpu"

    # Load model on XPU (matching production)
    print(f"\nLoading whisper model on device: {device}...")
    try:
        model = whisper.load_model("tiny", device=device)
        print(f"✓ Model loaded successfully on {device}")
        print(f"  Model device: {model.device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Test 1: File without audio (should fail)
    test_file_no_audio = Path(r"C:\Users\latch\connor_family_movies_processed\scenes\tape 1\tape 1-Scene-001.rife.1.realesrgan.mkv")

    # Test 2: File with audio (should work)
    test_file_with_audio = Path(r"C:\Users\latch\connor_family_movies_processed\scenes\tape 1\tape 1-Scene-001.mp4")

    for test_file in [test_file_no_audio, test_file_with_audio]:
        if not test_file.exists():
            print(f"\n⚠️  Test file not found: {test_file}")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {test_file.name}")
        print(f"{'='*70}")

        try:
            print("Calling model.transcribe()...")
            result = model.transcribe(str(test_file))
            print(f"✓ Transcription succeeded!")
            print(f"  Text: {result['text'][:100]}..." if len(result.get('text', '')) > 100 else f"  Text: {result.get('text', '')}")
            print(f"  Segments: {len(result.get('segments', []))}")
        except RuntimeError as e:
            print(f"✓ Caught RuntimeError (expected for files without audio):")
            error_str = str(e)
            if len(error_str) > 500:
                print(f"  {error_str[:250]}...")
                print(f"  ...{error_str[-250:]}")
            else:
                print(f"  {error_str}")
        except TypeError as e:
            print(f"✗ Caught TypeError (THIS IS THE BUG):")
            print(f"  Type: {type(e)}")
            print(f"  Message: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()

            # Check if the exception has a .shape attribute
            if hasattr(e, 'shape'):
                print(f"\n⚠️  TypeError object has .shape attribute: {e.shape}")
            else:
                print(f"\n  TypeError does NOT have .shape attribute")
        except Exception as e:
            print(f"✗ Caught unexpected exception: {type(e).__name__}")
            print(f"  Message: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    test_whisper_on_xpu()
