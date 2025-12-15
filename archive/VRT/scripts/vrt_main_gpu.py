"""
Modified VRT main_test_vrt.py to support Intel Arc GPUs via IPEX.
This script patches VRT to use Intel XPU when available.
"""
import sys
import os
from pathlib import Path

# Add VRT to path
VRT_PATH = Path(r"C:\Users\latch\VRT")
sys.path.insert(0, str(VRT_PATH))

# Try to import IPEX for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
    if ipex.xpu.is_available():
        print(f"Intel GPU (XPU) detected: {ipex.xpu.device_count()} device(s)")
        for i in range(ipex.xpu.device_count()):
            print(f"  Device {i}: {ipex.xpu.get_device_name(i)}")
    else:
        print("Intel GPU (XPU) not available, will use CPU")
        HAS_IPEX = False
except ImportError:
    print("Intel Extension for PyTorch not available, will use CPU")
    HAS_IPEX = False
    ipex = None

# Import VRT's main script
import main_test_vrt

# Patch the device selection in prepare_model_dataset
original_prepare = main_test_vrt.prepare_model_dataset

def patched_prepare_model_dataset(args):
    """Patched version that uses Intel XPU if available."""
    model = original_prepare(args)
    
    # Determine device
    if HAS_IPEX and ipex.xpu.is_available():
        device = torch.device('xpu:0')  # Use first Intel GPU
        print(f"Using Intel GPU (XPU) device: {device}")
        # Set environment for Intel GPU
        os.environ["ZE_AFFINITY_MASK"] = "0"  # Can be "0,1" for both GPUs
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU device: {device}")
    
    model.eval()
    model = model.to(device)
    
    # If using IPEX, optimize the model
    if HAS_IPEX and ipex.xpu.is_available():
        try:
            model = ipex.optimize(model, dtype=torch.float32)
            print("Model optimized for Intel GPU")
        except Exception as e:
            print(f"Warning: Could not optimize model for Intel GPU: {e}")
    
    return model

# Patch the main function's device selection
import torch
original_main = main_test_vrt.main

def patched_main():
    """Patched main that uses Intel XPU."""
    parser = main_test_vrt.argparse.ArgumentParser()
    # ... copy all arguments from original ...
    # For now, just call original but with device patching
    return original_main()

# Replace the function
main_test_vrt.prepare_model_dataset = patched_prepare_model_dataset

if __name__ == "__main__":
    # Import torch here after patching
    import torch
    main_test_vrt.main()

