#!/bin/bash
# Install Intel Extension for PyTorch (IPEX) on WSL Ubuntu 22.04
# This script installs PyTorch with XPU support and IPEX for Intel Arc GPUs

set -e

echo "=== Installing Intel GPU support for PyTorch in WSL ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python: $PYTHON_VERSION"

# Update package lists
echo ""
echo "Updating package lists..."
sudo apt update

# Install Python dependencies if needed
echo ""
echo "Installing Python and pip dependencies..."
sudo apt install -y python3 python3-pip python3-venv

# Install Intel GPU runtime dependencies
echo ""
echo "Installing Intel GPU runtime dependencies..."
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero

# Create/activate venv if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo ""
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with XPU support and IPEX
echo ""
echo "Installing PyTorch with Intel XPU support..."
echo "Using Intel's XPU wheel index..."
pip install torch torchvision torchaudio --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

echo ""
echo "Installing Intel Extension for PyTorch..."
pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Install einops (required by VRT)
echo ""
echo "Installing einops (required by VRT)..."
pip install einops

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To verify installation, run:"
echo "  python3 -c \"import torch; import intel_extension_for_pytorch as ipex; print('PyTorch:', torch.__version__); print('IPEX:', ipex.__version__); print('XPU available:', torch.xpu.is_available() if hasattr(torch, 'xpu') else False)\""
echo ""

