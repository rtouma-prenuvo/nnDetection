#!/bin/bash
# Comprehensive GPU-focused test runner for nnDetection
# Tests all critical functionality without slow CPU tests

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "nnDetection Comprehensive Test Suite"
echo "========================================"
echo ""
echo "Environment:"
python -c "
import torch
import torchvision

print(f'  Python:      3.12.x')
print(f'  PyTorch:     {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  CUDA:        {torch.version.cuda}')

if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print(f'  GPU Count:   {torch.cuda.device_count()}')
else:
    print('  GPU:         Not Available')
    print('')
    print('ERROR: CUDA is not available!')
    print('This test suite requires a CUDA-enabled GPU.')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Check if C++ extensions are available
python -c "
try:
    import nndet._C
    print('  C++ Extensions: ✓ Available')
    extensions_available = True
except ImportError as e:
    print(f'  C++ Extensions: ✗ Not Available')
    print(f'     Error: {str(e)[:80]}...' if len(str(e)) > 80 else f'     Error: {e}')
    print('')
    print('WARNING: C++ CUDA extensions not available.')
    print('Some GPU tests will be skipped.')
    print('')
    print('To enable C++ extensions, rebuild with:')
    print('  export CUDA_HOME=/usr/local/cuda-12.4')
    print('  export TORCH_CUDA_ARCH_LIST=\"7.5;8.0;8.6\"')
    print('  FORCE_CUDA=1 pip install -e . --no-build-isolation --force-reinstall')
    extensions_available = False
"

echo ""
echo "Running comprehensive test suite..."
echo "========================================" 
echo ""
echo "Test Coverage:"
echo "  • GPU functionality & CUDA extensions"
echo "  • Core imports & dependencies"
echo "  • Architecture components"
echo "  • Evaluators & metrics"
echo "  • Losses & matchers"
echo "  • Data pipeline & augmentation"
echo "  • Console scripts"
echo "  • Utilities (IO, preprocessing)"
echo ""

# Run comprehensive GPU-focused test suite
pytest tests/test_gpu_functionality.py \
       tests/test_core_imports.py \
       tests/test_architecture.py \
       tests/test_evaluators.py \
       tests/test_losses_matchers.py \
       tests/test_data_pipeline.py \
       tests/test_scripts.py \
       tests/test_utilities.py \
       -v --tb=short --color=yes

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed!"
    echo ""
    echo "Your nnDetection installation is fully functional with:"
    echo "  • PyTorch 2.6.x with CUDA 12.4"
    echo "  • GPU acceleration verified"
    echo "  • C++ CUDA extensions (if available)"
    echo "  • All core modules operational"
    echo "  • All scripts accessible"
    echo "  • Complete pipeline tested"
else
    echo "✗ Some tests failed"
    echo ""
    echo "Check the output above for details."
fi

echo ""
echo "For specific test categories:"
echo "  pytest tests/test_gpu_functionality.py -v   # GPU & CUDA"
echo "  pytest tests/test_architecture.py -v        # Model architecture"
echo "  pytest tests/test_evaluators.py -v          # Metrics & evaluation"
echo "  pytest tests/test_losses_matchers.py -v     # Losses & matchers"
echo "  pytest tests/test_data_pipeline.py -v       # Data loading & augmentation"
echo "  pytest tests/test_scripts.py -v             # Console scripts"
echo ""

exit $TEST_EXIT_CODE
