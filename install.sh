#!/bin/bash
# Automated installation script for nnDetection with C++ CUDA extensions
# For CUDA 12.4 + PyTorch 2.6

set -e

echo "================================"
echo "nnDetection Installation Script"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the nnDetection directory"
    exit 1
fi

# Step 1: Install PyTorch 2.6 with CUDA 12.4
echo "Step 1: Installing PyTorch 2.6.0 with CUDA 12.4..."
pip install -r requirements-cu124.txt

echo ""
echo "Step 2: Setting up CUDA environment..."

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# GPU architecture - adjust if needed
# 7.5 = Tesla T4, RTX 2080
# 8.0 = A100
# 8.6 = RTX 3090, A6000, A40
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Verify CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc not found. Make sure CUDA toolkit 12.4 is installed."
    echo "C++ extensions will not be built."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    # Install without CUDA extensions
    echo ""
    echo "Step 3: Installing nnDetection (without C++ extensions)..."
    pip install -e .
else
    echo "CUDA found: $(nvcc --version | grep release)"
    
    # Clean previous builds
    echo ""
    echo "Cleaning previous builds..."
    rm -rf build nndet.egg-info nndet/_C*.so
    
    # Build with CUDA extensions
    echo ""
    echo "Step 3: Installing nnDetection with C++ CUDA extensions..."
    FORCE_CUDA=1 pip install -e . --no-build-isolation
fi

# Verify installation
echo ""
echo "================================"
echo "Verifying Installation"
echo "================================"

python -c "
import torch
import torchvision
import nndet

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ torchvision: {torchvision.__version__}')
print(f'✓ nnDetection: {nndet.__version__ if hasattr(nndet, \"__version__\") else \"0.2.0\"}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

try:
    import nndet._C
    print('✓ C++ CUDA extensions: Available')
except ImportError as e:
    print(f'⚠ C++ CUDA extensions: Not available ({e})')
    print('  (3D GPU NMS will fall back to slower CPU implementation)')
"

echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "To test the installation, run:"
echo "  ./run_tests.sh"
echo ""

