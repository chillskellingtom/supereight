# Installing Intel Extension for PyTorch (IPEX) for Intel Arc GPUs

## Summary
✅ **IPEX is installed** but needs oneAPI runtime libraries in PATH  
✅ **VRT is patched** to use Intel GPUs automatically  
✅ **Models work directly** - no conversion needed  

## Current Status
- IPEX 2.1.40+xpu installed for Python 3.11
- PyTorch 2.9.1 installed
- oneAPI Base Toolkit installed at `C:\Program Files (x86)\Intel\oneAPI`
- **Issue**: Missing DLL dependencies (likely PATH or runtime libraries)

## Solution: Set oneAPI Environment

The IPEX DLL requires oneAPI runtime libraries. You need to initialize the oneAPI environment before running VRT:

### Option 1: Use the Setup Script (Recommended)
```powershell
# Run VRT with Intel GPU support
..\\scripts\\setup_intel_gpu_env.ps1
```

### Option 2: Manual Setup
Before running VRT, initialize oneAPI:
```powershell
# Initialize oneAPI environment
& "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

# Set Intel GPU variables
$env:ZE_AFFINITY_MASK = "0,1"  # Use both GPUs
$env:SYCL_CACHE_PERSISTENT = "1"

# Now run VRT
python scripts/vrt_enhance.py --task videosr_reds_16frames ...
```

### Option 3: Update scripts/vrt_enhance.py to Auto-Initialize
The script can be updated to automatically initialize oneAPI before running.

## Installation Commands (Already Done)
```bash
# PyTorch with Intel GPU support
pip install torch torchvision torchaudio --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Intel Extension for PyTorch
pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## Verification
After setting oneAPI environment, test with:
```python
python -c "import intel_extension_for_pytorch as ipex; print('XPU:', ipex.xpu.is_available()); print('Devices:', ipex.xpu.device_count())"
```

## Next Steps
1. Update `scripts/vrt_enhance.py` to automatically initialize oneAPI
2. Or use `setup_intel_gpu_env.ps1` before running VRT
3. Test with a small scene first to verify GPU acceleration

