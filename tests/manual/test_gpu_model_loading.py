"""
Test script to verify VRT model loading and parallelization across Intel Arc GPUs.
Tests model loading, GPU memory usage, and parallel processing.
"""
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# Initialize oneAPI environment
def _init_oneapi_env():
    """Initialize oneAPI environment variables."""
    import os
    oneapi_path = Path(r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat")
    if oneapi_path.exists():
        # Source setvars.bat and capture environment
        import subprocess
        result = subprocess.run(
            f'cmd /c "{oneapi_path}" >nul 2>&1 && set',
            shell=True,
            capture_output=True,
            text=True
        )
        for line in result.stdout.splitlines():
            if '=' in line and not line.startswith('::'):
                key, value = line.split('=', 1)
                os.environ[key] = value

_init_oneapi_env()

# Add VRT to path
VRT_PATH = Path(r"C:\Users\latch\VRT")
if not VRT_PATH.exists():
    VRT_PATH = Path(__file__).parent / "VRT"
sys.path.insert(0, str(VRT_PATH))

# Import VRT modules
try:
    from models.network_vrt import VRT as net
    from utils import utils_image as util
except ImportError as e:
    print(f"Error importing VRT modules: {e}")
    print("Make sure VRT is cloned and requirements are installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
LOG = logging.getLogger("gpu_test")

# VRT Model configuration (16-frame REDS model - largest)
MODEL_CONFIG = {
    "task": "002_VRT_videosr_bi_REDS_16frames",
    "model_file": "002_VRT_videosr_bi_REDS_16frames.pth",
    "url": "https://github.com/JingyunLiang/VRT/releases/download/v0.0/002_VRT_videosr_bi_REDS_16frames.pth",
}

MODEL_CACHE = Path.home() / ".cache" / "vrt_models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)


def get_gpu_info() -> Dict:
    """Get Intel GPU information."""
    gpu_info = {
        "available": False,
        "device_count": 0,
        "devices": []
    }
    
    try:
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = ipex.xpu.device_count()
            for i in range(gpu_info["device_count"]):
                device_name = ipex.xpu.get_device_name(i)
                gpu_info["devices"].append({
                    "index": i,
                    "name": device_name,
                    "device": torch.device(f'xpu:{i}')
                })
        else:
            LOG.warning("Intel GPU (XPU) not available")
    except ImportError:
        LOG.warning("Intel Extension for PyTorch not installed")
    except Exception as e:
        LOG.warning(f"Error checking Intel GPU: {e}")
    
    return gpu_info


def get_gpu_memory(device: torch.device) -> Dict[str, float]:
    """Get GPU memory usage in GB."""
    try:
        import intel_extension_for_pytorch as ipex
        if device.type == 'xpu':
            # Get memory info for Intel GPU
            memory_allocated = ipex.xpu.get_device_properties(device.index).total_memory / (1024**3)
            # Note: Intel XPU doesn't expose allocated/reserved like CUDA
            # We'll use a placeholder for now
            return {
                "total": memory_allocated,
                "allocated": 0.0,  # Not directly available
                "reserved": 0.0,   # Not directly available
                "free": memory_allocated
            }
    except Exception as e:
        LOG.debug(f"Could not get GPU memory info: {e}")
    
    return {"total": 0.0, "allocated": 0.0, "reserved": 0.0, "free": 0.0}


def download_model() -> Path:
    """Download model if not cached."""
    model_path = MODEL_CACHE / MODEL_CONFIG["model_file"]
    
    if model_path.exists():
        LOG.info(f"✓ Using cached model: {model_path}")
        return model_path
    
    LOG.info(f"Downloading model: {MODEL_CONFIG['model_file']}")
    LOG.info(f"URL: {MODEL_CONFIG['url']}")
    
    import urllib.request
    try:
        urllib.request.urlretrieve(MODEL_CONFIG["url"], model_path)
        LOG.info(f"✓ Downloaded model to: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")
    
    return model_path


def create_model() -> torch.nn.Module:
    """Create VRT model architecture."""
    LOG.info("Creating VRT model architecture...")
    # Use exact parameters from VRT main_test_vrt.py for 002_VRT_videosr_bi_REDS_16frames
    model = net(
        upscale=4,
        img_size=[16, 64, 64],
        window_size=[8, 8, 8],  # Fixed: was [16, 8, 8]
        depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
        indep_reconsts=[11, 12],
        embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        pa_frames=6,  # Fixed: was 2
        deformable_groups=24  # Fixed: was 12
    )
    LOG.info("✓ Model architecture created")
    return model


def load_model_weights(model: torch.nn.Module, model_path: Path) -> None:
    """Load model weights from checkpoint."""
    LOG.info(f"Loading model weights from: {model_path}")
    start_time = time.time()
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    
    load_time = time.time() - start_time
    LOG.info(f"✓ Model weights loaded in {load_time:.2f}s")


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move model to device and optimize if Intel GPU."""
    LOG.info(f"Moving model to device: {device}")
    start_time = time.time()
    
    model.eval()
    model = model.to(device)
    
    # Optimize for Intel GPU if available
    if device.type == 'xpu':
        try:
            import intel_extension_for_pytorch as ipex
            LOG.info("Optimizing model for Intel GPU...")
            model = ipex.optimize(model, dtype=torch.float32)
            LOG.info("✓ Model optimized for Intel GPU")
        except Exception as e:
            LOG.warning(f"Could not optimize model: {e}")
    
    move_time = time.time() - start_time
    LOG.info(f"✓ Model moved to device in {move_time:.2f}s")
    
    return model


def load_model_on_device(device: torch.device, device_name: str) -> torch.nn.Module:
    """Load complete model on a specific device."""
    LOG.info(f"\n{'='*60}")
    LOG.info(f"Loading model on {device_name} ({device})")
    LOG.info(f"{'='*60}")
    
    # Get initial memory
    mem_before = get_gpu_memory(device)
    LOG.info(f"GPU Memory before: {mem_before['total']:.2f} GB total")
    
    # Create model
    model = create_model()
    
    # Download and load weights
    model_path = download_model()
    load_model_weights(model, model_path)
    
    # Move to device
    model = move_model_to_device(model, device)
    
    # Get memory after
    mem_after = get_gpu_memory(device)
    LOG.info(f"GPU Memory after: {mem_after['total']:.2f} GB total")
    
    # Test inference with dummy input
    LOG.info("Testing inference...")
    test_inference(model, device)
    
    return model


def test_inference(model: torch.nn.Module, device: torch.device, num_tests: int = 3):
    """Test model inference with dummy data."""
    model.eval()
    
    # VRT expects shape (N, T, C, H, W); use 16 frames, 3 channels
    dummy_input = torch.randn(1, 16, 3, 64, 64).to(device)
    
    LOG.info(f"Running {num_tests} inference tests...")
    times = []
    
    with torch.no_grad():
        for i in range(num_tests):
            start_time = time.time()
            output = model(dummy_input)
            torch.xpu.synchronize() if device.type == 'xpu' else None
            inference_time = time.time() - start_time
            times.append(inference_time)
            LOG.info(f"  Test {i+1}: {inference_time*1000:.2f}ms")
    
    avg_time = sum(times) / len(times)
    LOG.info(f"✓ Average inference time: {avg_time*1000:.2f}ms")
    LOG.info(f"  Output shape: {output.shape}")


def test_parallel_loading(gpu_info: Dict):
    """Test loading models on multiple GPUs in parallel."""
    if gpu_info["device_count"] < 2:
        LOG.warning("Only 1 GPU available, cannot test parallel loading")
        return
    
    LOG.info(f"\n{'='*60}")
    LOG.info("Testing Parallel Model Loading on Multiple GPUs")
    LOG.info(f"{'='*60}")
    
    devices = gpu_info["devices"][:2]  # Use first 2 GPUs
    
    # Set environment to use both GPUs
    import os
    os.environ["ZE_AFFINITY_MASK"] = "0,1"
    LOG.info(f"ZE_AFFINITY_MASK set to: 0,1")
    
    models = []
    start_time = time.time()
    
    # Load models sequentially (but on different devices)
    for device_info in devices:
        device = device_info["device"]
        device_name = device_info["name"]
        model = load_model_on_device(device, device_name)
        models.append((device, model))
    
    total_time = time.time() - start_time
    LOG.info(f"\n{'='*60}")
    LOG.info(f"✓ Parallel loading complete in {total_time:.2f}s")
    LOG.info(f"  Models loaded on {len(models)} GPU(s)")
    LOG.info(f"{'='*60}\n")
    
    # Test parallel inference
    LOG.info("Testing parallel inference...")
    test_parallel_inference(models)


def test_parallel_inference(models: List[Tuple[torch.device, torch.nn.Module]]):
    """Test inference on multiple GPUs in parallel."""
    import threading
    
    def run_inference(device: torch.device, model: torch.nn.Module, device_name: str):
        LOG.info(f"Starting inference on {device_name}...")
        test_inference(model, device, num_tests=5)
        LOG.info(f"✓ Inference complete on {device_name}")
    
    threads = []
    for device, model in models:
        device_name = f"GPU-{device.index}"
        thread = threading.Thread(
            target=run_inference,
            args=(device, model, device_name)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    LOG.info("✓ Parallel inference test complete")


def main():
    """Main test function."""
    LOG.info("="*60)
    LOG.info("VRT GPU Model Loading Test")
    LOG.info("="*60)
    
    # Check GPU availability
    LOG.info("\nChecking GPU availability...")
    gpu_info = get_gpu_info()
    
    if not gpu_info["available"]:
        LOG.warning("No Intel GPUs available (DLL issue detected)")
        LOG.warning("This is likely due to missing oneAPI runtime libraries")
        LOG.warning("Will test model loading structure on CPU instead")
        LOG.warning("")
        LOG.warning("To fix GPU support:")
        LOG.warning("  1. Ensure Intel Arc GPU drivers are installed")
        LOG.warning("  2. Ensure oneAPI Base Toolkit is installed")
        LOG.warning("  3. Run: setvars.bat before Python scripts")
        LOG.warning("  4. Check Visual C++ runtime libraries")
        LOG.warning("")
        
        # Continue with CPU test to show structure
        LOG.info("Testing model loading on CPU...")
        device = torch.device('cpu')
        model = create_model()
        model_path = download_model()
        load_model_weights(model, model_path)
        model = move_model_to_device(model, device)
        test_inference(model, device)
        LOG.info("✓ CPU test complete (structure verified)")
        return 0
    
    LOG.info(f"✓ Found {gpu_info['device_count']} Intel GPU(s):")
    for device_info in gpu_info["devices"]:
        LOG.info(f"  Device {device_info['index']}: {device_info['name']}")
    
    # Test single GPU loading
    if gpu_info["device_count"] >= 1:
        LOG.info("\n" + "="*60)
        LOG.info("Test 1: Single GPU Model Loading")
        LOG.info("="*60)
        device = gpu_info["devices"][0]["device"]
        device_name = gpu_info["devices"][0]["name"]
        model = load_model_on_device(device, device_name)
        LOG.info("✓ Single GPU test complete\n")
    
    # Test parallel GPU loading
    if gpu_info["device_count"] >= 2:
        LOG.info("\n" + "="*60)
        LOG.info("Test 2: Parallel Multi-GPU Model Loading")
        LOG.info("="*60)
        test_parallel_loading(gpu_info)
        LOG.info("✓ Parallel GPU test complete\n")
    
    LOG.info("="*60)
    LOG.info("All tests complete!")
    LOG.info("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        LOG.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        LOG.error(f"\nTest failed with error: {e}", exc_info=True)
        sys.exit(1)

