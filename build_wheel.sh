#!/bin/bash
# Build wheel for nnDetection with C++ CUDA extensions
# For CUDA 12.4 and PyTorch 2.6.0+cu124

set -e

# Read version from pyproject.toml dynamically
VERSION=$(grep -E '^version\s*=' pyproject.toml | sed -E 's/.*version\s*=\s*"([^"]+)".*/\1/')
if [ -z "$VERSION" ]; then
    echo "ERROR: Could not determine version from pyproject.toml"
    exit 1
fi

CUDA_VERSION="cu124"
PYTHON_VERSION="cp312"

echo "========================================"
echo "nnDetection Wheel Build Script"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Package version: ${VERSION}"
echo "  CUDA version: ${CUDA_VERSION}"
echo "  Python version: ${PYTHON_VERSION}"
echo ""

# Change to nnDetection directory and save absolute path
cd "$(dirname "$0")"
NNDET_DIR="$(pwd)"

# Verify we have the right PyTorch version
echo "Verifying PyTorch installation..."
python -c "
import torch
import sys

expected_cuda = '12.4'
actual_cuda = torch.version.cuda

if actual_cuda != expected_cuda:
    print(f'ERROR: PyTorch CUDA version mismatch!')
    print(f'Expected: {expected_cuda}, Got: {actual_cuda}')
    print(f'')
    print('Install correct PyTorch version:')
    print('  pip install -r requirements-cu124.txt')
    sys.exit(1)

print(f'✓ PyTorch {torch.__version__} (CUDA {actual_cuda})')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Set CUDA environment variables
echo ""
echo "Setting CUDA environment..."
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# GPU architectures to support
# 7.5 = Tesla T4, RTX 2080
# 8.0 = A100
# 8.6 = RTX 3090, A6000, A40  
# 8.9 = RTX 4090, H100, L4, L40
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Verify CUDA toolkit
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found!"
    echo "Make sure CUDA toolkit 12.4 is installed at $CUDA_HOME"
    exit 1
fi

echo "✓ CUDA toolkit: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "✓ GPU architectures: ${TORCH_CUDA_ARCH_LIST}"

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info nndet/_C*.so
echo "✓ Clean complete"

# Build wheel with CUDA extensions
echo ""
echo "Building wheel with C++ CUDA extensions..."
echo "(This may take a few minutes...)"
FORCE_CUDA=1 python setup.py bdist_wheel

# Check if build succeeded
if [ ! -d "dist" ] || [ -z "$(ls -A dist/*.whl 2>/dev/null)" ]; then
    echo "ERROR: Wheel build failed!"
    exit 1
fi

# Rename wheel to include CUDA version
echo ""
echo "Renaming wheel to include CUDA version..."
cd dist
WHEEL_NAME=""
# Escape dots in version for sed pattern matching
VERSION_ESCAPED=$(echo "$VERSION" | sed 's/\./\\./g')
# Try to find wheel matching version first, then any nndet wheel
for wheel in nndet-${VERSION}-*.whl nndet-*.whl; do
    if [ -f "$wheel" ]; then
        # Insert +cu124 after version number (before the first hyphen after version)
        new_name=$(echo "$wheel" | sed "s/\(${VERSION_ESCAPED}\)\(-[^+]*\.whl\)/\1+${CUDA_VERSION}\2/")
        mv "$wheel" "$new_name"
        echo "✓ Created: $new_name"
        
        # Store wheel name for later
        WHEEL_NAME="$new_name"
        break
    fi
done
cd ..

# Check if wheel was found
if [ -z "$WHEEL_NAME" ]; then
    echo "ERROR: No wheel file found in dist/ directory!"
    exit 1
fi

# Verify wheel contents
echo ""
echo "Verifying wheel contents..."
python -c "
import zipfile
import sys
import os

wheel_path = 'dist/${WHEEL_NAME}'

if not os.path.exists(wheel_path):
    print(f'ERROR: Wheel file not found: {wheel_path}')
    sys.exit(1)

if not os.path.isfile(wheel_path):
    print(f'ERROR: Path is not a file: {wheel_path}')
    sys.exit(1)

with zipfile.ZipFile(wheel_path, 'r') as zf:
    files = zf.namelist()
    
    # Check for C++ extension
    so_files = [f for f in files if f.endswith('.so')]
    
    if so_files:
        print(f'✓ C++ extension found: {so_files[0]}')
    else:
        print('WARNING: No C++ extension (.so) found in wheel!')
        print('The wheel was built without CUDA extensions.')
    
    # Check for Python files
    py_files = [f for f in files if f.endswith('.py') and 'nndet/' in f]
    print(f'✓ Python files: {len(py_files)} files')
    
    # Check for metadata
    metadata = [f for f in files if 'METADATA' in f]
    if metadata:
        print(f'✓ Metadata: {metadata[0]}')
"

# Test the wheel
echo ""
echo "Testing wheel installation in temporary environment..."

# Create test directory away from source to avoid import conflicts
TEST_DIR="/tmp/test_nndet_wheel_$(date +%s)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

python -m venv venv
source venv/bin/activate

# Install PyTorch first
pip install -q torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install our wheel (use saved path from earlier)
WHEEL_PATH="${NNDET_DIR}/dist/${WHEEL_NAME}"
pip install -q "$WHEEL_PATH"

# Test import and C++ extensions (run from test dir, not source dir)
python -c "
import sys
import torch
import nndet

# Verify we're loading from site-packages, not source
import os
nndet_path = os.path.dirname(nndet.__file__)
if 'site-packages' not in nndet_path:
    print(f'ERROR: Loading from source ({nndet_path}), not from wheel!')
    exit(1)

print(f'✓ nndet loaded from: {nndet_path}')

try:
    import nndet._C
    print('✓ C++ CUDA extensions available')
except ImportError as e:
    print(f'✗ C++ extensions not available: {e}')
    # Check if .so file exists
    so_path = os.path.join(nndet_path, '_C.cpython-312-x86_64-linux-gnu.so')
    if os.path.exists(so_path):
        print(f'  Note: .so file exists at {so_path}')
        print(f'  This is likely a runtime library path issue')
    exit(1)

print(f'✓ Wheel test passed')
"

TEST_RESULT=$?
deactivate
cd - > /dev/null
rm -rf "$TEST_DIR"

if [ $TEST_RESULT -ne 0 ]; then
    echo ""
    echo "ERROR: Wheel test failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Wheel Build Complete!"
echo "========================================"
echo ""
echo "Wheel location: dist/${WHEEL_NAME}"
echo ""
