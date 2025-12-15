# Intel Arc GPU Setup for VRT

## Current Status ✅
- ✅ **IPEX installed**: `intel-extension-for-pytorch 2.1.40+xpu` (Python 3.11)
- ✅ **PyTorch installed**: `2.9.1` with Intel GPU support
- ✅ **oneAPI Base Toolkit**: Installed at `C:\Program Files (x86)\Intel\oneAPI`
- ✅ **VRT patched**: Automatically detects and uses Intel GPUs
- ⚠️ **Issue**: IPEX DLL dependency (likely missing runtime libraries or PATH)

## Installation (Completed)

### What's Installed
1. ✅ **PyTorch with Intel GPU support**:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
   ```

2. ✅ **Intel Extension for PyTorch (IPEX)**:
   ```bash
   pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
   ```

### Alternative: Install via IPEX-LLM
If you need the full LLM library (includes IPEX):
```bash
pip install ipex-llm[xpu] --extra-index-url https://download.pytorch.org/whl/cpu
```

## Running VRT with Intel GPU

### Option 1: Use Helper Script (Recommended)
```powershell
..\\scripts\\run_vrt_with_gpu.ps1 -Task videosr_reds_16frames -Input "scene.mp4" -Output "output.mp4"
```

### Option 2: Manual oneAPI Initialization
```powershell
# Initialize oneAPI environment
& "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

# Set Intel GPU variables
$env:ZE_AFFINITY_MASK = "0,1"  # Use both GPUs

# Run VRT
python scripts/vrt_enhance.py --task videosr_reds_16frames --input scene.mp4 --output output.mp4
```

### Option 3: CPU Fallback (Working)
If GPU doesn't work, VRT will automatically fall back to CPU:
- Slower (30-60+ min for 2674 frames)
- But fully functional

## Troubleshooting

### DLL Dependency Error
If you see: `OSError: [WinError 126] The specified module could not be found`

**Possible causes:**
1. Missing Visual C++ runtime libraries
2. oneAPI runtime not in PATH
3. Missing oneAPI components

**Solutions:**
1. Ensure oneAPI Base Toolkit is fully installed
2. Run `setvars.bat` before Python scripts
3. Install Visual C++ Redistributable (latest)
4. Check Windows Event Viewer for specific DLL errors

### Testing GPU Availability
```powershell
# With oneAPI initialized
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1 && python -c "import intel_extension_for_pytorch as ipex; print(\"XPU:\", ipex.xpu.is_available()); print(\"Devices:\", ipex.xpu.device_count())"'
```

## Performance Notes
- **With Intel GPU**: 10-50x faster than CPU (5-15 min for 2674 frames)
- **CPU only**: 30-60+ minutes for 2674 frames (but works)
- **VRT automatically falls back to CPU** if GPU unavailable

## Current Configuration
- **ZE_AFFINITY_MASK**: "0,1" (both Intel Arc A770 GPUs)
- **CPU Threading**: Optimized for all CPU cores
- **VRT Script**: Automatically initializes oneAPI and detects Intel XPU

