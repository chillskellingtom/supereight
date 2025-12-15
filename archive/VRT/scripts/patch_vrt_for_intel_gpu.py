"""
Patch VRT's main_test_vrt.py to use Intel Arc GPUs.
This modifies the device selection to use Intel XPU when available.
"""
import shutil
from pathlib import Path

VRT_MAIN = Path(r"C:\Users\latch\VRT\main_test_vrt.py")
BACKUP = VRT_MAIN.with_suffix(".py.backup")

if not VRT_MAIN.exists():
    print(f"Error: {VRT_MAIN} not found")
    exit(1)

# Backup original
if not BACKUP.exists():
    shutil.copy2(VRT_MAIN, BACKUP)
    print(f"Backed up original to {BACKUP}")

# Read the file
with open(VRT_MAIN, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already patched
if "intel_extension_for_pytorch" in content or "xpu" in content.lower():
    print("File already appears to be patched")
    exit(0)

# Find the device selection line and replace it
old_line = "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
new_code = """    # Check for Intel GPU (XPU) support
    device = torch.device('cpu')
    try:
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            device = torch.device('xpu:0')
            print(f'Using Intel GPU (XPU): {ipex.xpu.get_device_name(0)}')
            # Set environment for Intel GPU
            import os
            os.environ['ZE_AFFINITY_MASK'] = '0'  # Use first GPU, '0,1' for both
        else:
            print('Intel GPU (XPU) not available, using CPU')
    except ImportError:
        # Fall back to CUDA or CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using CUDA')
        else:
            device = torch.device('cpu')
            print('Using CPU (Intel GPU support not available)')
    except Exception as e:
        print(f'Error checking Intel GPU: {e}, using CPU')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')"""

if old_line in content:
    content = content.replace(old_line, new_code)
    
    # Also patch model.to(device) to handle XPU optimization
    model_to_device = "    model = model.to(device)"
    optimized_code = """    model = model.to(device)
    # Optimize for Intel GPU if using XPU
    if device.type == 'xpu':
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.float32)
            print('Model optimized for Intel GPU')
        except Exception as e:
            print(f'Warning: Could not optimize for Intel GPU: {e}')"""
    
    if model_to_device in content:
        content = content.replace(model_to_device, optimized_code)
    
    # Write patched file
    with open(VRT_MAIN, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Patched {VRT_MAIN} for Intel GPU support")
    print("Note: Install Intel Extension for PyTorch with:")
    print("  pip install intel-extension-for-pytorch")
else:
    print("Could not find device selection line to patch")
    print("File structure may have changed")

