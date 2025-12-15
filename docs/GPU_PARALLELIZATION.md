# VRT GPU Parallelization and Model Loading

## Overview

This document explains how VRT models are loaded and parallelized across Intel Arc GPUs, with comprehensive logging and progress reporting.

## Current Status

### ✅ Implemented Features

1. **Multi-GPU Detection**: Automatically detects Intel Arc GPUs (XPU devices)
2. **Environment Configuration**: Sets `ZE_AFFINITY_MASK=0,1` to use both GPUs
3. **Model Loading Logging**: Detailed logs for:
   - Model architecture creation
   - Model weight loading (with file size)
   - Device placement (CPU/XPU)
   - Model optimization (IPEX)
4. **Progress Monitoring**: Real-time progress tracking during VRT inference
5. **GPU Memory Reporting**: Logs GPU memory usage (when available)

### ⚠️ Current Limitations

**DLL Dependency Issue**: Intel Extension for PyTorch (IPEX) requires oneAPI runtime libraries that may not be properly initialized. This causes:
- Models to fall back to CPU processing
- GPU acceleration not working despite GPUs being detected

**Workaround**: Ensure `setvars.bat` is sourced before running Python scripts (handled by `run_vrt_with_gpu.ps1`).

## How Multi-GPU Parallelization Works

### 1. GPU Detection

```python
import intel_extension_for_pytorch as ipex
if ipex.xpu.is_available():
    device_count = ipex.xpu.device_count()  # Should be 2 for dual A770
    for i in range(device_count):
        device_name = ipex.xpu.get_device_name(i)
```

### 2. Environment Setup

```python
env["ZE_AFFINITY_MASK"] = "0,1"  # Use both GPUs
env["SYCL_CACHE_PERSISTENT"] = "1"
env["SYCL_CACHE_DIR"] = str(Path.home() / ".cache" / "intel_gpu_cache")
```

### 3. Model Loading (Per GPU)

When processing videos, VRT loads models as follows:

1. **Model Architecture Creation**: Creates VRT network structure
2. **Weight Loading**: Loads `.pth` checkpoint file
3. **Device Placement**: Moves model to GPU (`xpu:0` or `xpu:1`)
4. **Optimization**: Applies IPEX optimization for Intel GPUs

### 4. Parallel Processing Strategy

**Current Implementation**: VRT's `main_test_vrt.py` uses a single model instance. For true multi-GPU parallelization, we would need:

- **Data Parallelism**: Split video frames across GPUs
- **Model Parallelism**: Split model layers across GPUs (complex)
- **Pipeline Parallelism**: Process different video segments on different GPUs

**Current Approach**: VRT uses `ZE_AFFINITY_MASK=0,1` which allows Level Zero to distribute work across both GPUs automatically, but the model itself runs on one device.

## Logging and Progress Reporting

### Model Loading Logs

```
[INFO] Intel GPU (XPU) available: 2 device(s)
[INFO]   Device 0: Intel(R) Arc(TM) A770 Graphics
[INFO]   Device 1: Intel(R) Arc(TM) A770 Graphics
[INFO] Model: 002_VRT_videosr_bi_REDS_16frames.pth (1234.56 MB)
[INFO] Loading model weights from: C:\Users\...\002_VRT_videosr_bi_REDS_16frames.pth
[INFO] ✓ Model weights loaded in 0.25s
[INFO] Moving model to device: xpu:0
[INFO] Optimizing model for Intel GPU...
[INFO] ✓ Model optimized for Intel GPU
[INFO] ✓ Model moved to device in 0.15s
```

### Processing Progress Logs

```
[INFO] ============================================================
[INFO] Starting VRT inference...
[INFO] ============================================================
[INFO] Processing 2674 frames...
[INFO] Progress: 50/2674 frames (1.9%)
[INFO] Progress: 134/2674 frames (5.0%)
[INFO] Progress: 268/2674 frames (10.0%)
...
[INFO] ============================================================
[INFO] ✓ VRT inference complete
[INFO] ============================================================
```

### GPU Memory Logs

```
[INFO] GPU Memory before: 16.00 GB total
[INFO] GPU Memory after: 15.23 GB total
[INFO] Memory used: 0.77 GB
```

## Testing GPU Model Loading

Run the test script to verify model loading:

```powershell
..\\scripts\\test_gpu_loading.ps1
```

This will:
1. Detect available GPUs
2. Load model architecture
3. Download model weights (if needed)
4. Load weights into model
5. Move model to GPU
6. Test inference
7. Report GPU memory usage

## Future Enhancements

### True Multi-GPU Parallelization

To achieve true parallelization across both GPUs, we could:

1. **Frame-Level Parallelism**:
   ```python
   # Split frames across GPUs
   frames_per_gpu = total_frames // num_gpus
   for gpu_idx, gpu_device in enumerate(gpus):
       gpu_frames = frames[gpu_idx * frames_per_gpu:(gpu_idx + 1) * frames_per_gpu]
       process_on_gpu(gpu_frames, gpu_device)
   ```

2. **Model Replication**:
   ```python
   # Load same model on each GPU
   models = []
   for gpu_device in gpus:
       model = load_model()
       model = model.to(gpu_device)
       models.append(model)
   ```

3. **Data Parallel Processing**:
   ```python
   # Use PyTorch DataParallel or DistributedDataParallel
   model = torch.nn.DataParallel(model, device_ids=[0, 1])
   ```

## Troubleshooting

### GPU Not Detected

1. Check oneAPI installation: `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
2. Verify IPEX installation: `pip list | findstr intel-extension-for-pytorch`
3. Check GPU drivers: Device Manager → Display adapters

### DLL Error

```
OSError: [WinError 126] The specified module could not be found.
Error loading "...intel-ext-pt-gpu.dll" or one of its dependencies.
```

**Solution**: Ensure `setvars.bat` is sourced before running Python:
```powershell
..\\scripts\\run_vrt_with_gpu.ps1 -Task videosr_reds_16frames -Input <video> -Output <output>
```

### Model Loading Slow

- First load is slow (model download + initialization)
- Subsequent loads are faster (cached model)
- GPU optimization adds ~0.1-0.5s overhead

### Memory Issues

- VRT models are large (~1-2 GB)
- Each GPU has 16 GB VRAM
- Should handle multiple models simultaneously
- Monitor with: `tests/manual/test_gpu_model_loading.py`

## Summary

The current implementation:
- ✅ Detects and configures multiple Intel GPUs
- ✅ Loads models with detailed logging
- ✅ Reports progress during processing
- ✅ Falls back gracefully to CPU if GPU unavailable
- ⚠️ Requires oneAPI environment initialization
- 🔄 Uses Level Zero for automatic GPU distribution (single model instance)

For true multi-GPU parallelization, additional code changes would be needed to explicitly split work across GPUs.

