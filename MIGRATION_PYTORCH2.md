# nnDetection v0.2 - PyTorch 2.x Migration Guide

## Table of Contents

- [What Changed](#what-changed)
- [Requirements](#requirements)
- [Installation](#installation)
  - [CUDA Toolkit Setup](#cuda-toolkit-setup)
  - [Install PyTorch 2.6](#install-pytorch-26)
  - [Build nnDetection](#build-nndetection)
  - [Verification](#verification)
- [Code Changes Reference](#code-changes-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## What Changed

### Framework Updates

| Component | Before | After |
|-----------|--------|-------|
| **PyTorch** | 1.10-1.11 | 2.6+ |
| **Lightning** | pytorch_lightning 1.3-1.4 | lightning 2.0+ |
| **Python** | 3.8+ | 3.9-3.12 |
| **CUDA** | 10.1-11.3 | 12.4+ |
| **torchmetrics** | 0.7.0-0.7.3 | 0.11.0+ |
| **SimpleITK** | <2.1.0 | >=2.1.0 |

---

## Installation

For automated installation

```bash
cd /path/to/nnDetection
./install.sh
```

### CUDA Toolkit Setup

The CUDA **toolkit** (including `nvcc` compiler) is required to build CUDA extensions.

#### Check if Already Installed

```bash
nvcc --version
```

If `nvcc` is found, skip to [Install PyTorch 2.6](#install-pytorch-26).

#### Install CUDA Toolkit 12.4

**For Ubuntu 22.04:**

```bash
# Download the CUDA GPG key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-4

# Set environment variables (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload environment
source ~/.bashrc
```

**For other distributions:** See [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

#### Verify Installation

```bash
nvcc --version
# Should show: Cuda compilation tools, release 12.4
```

### Install PyTorch 2.6

**Option A: Using requirements file (recommended)**
```bash
pip install -r requirements-cu124.txt
```

**Option B: Manual installation**
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

**Verify PyTorch:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Build nnDetection

#### Set Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  # Adjust for your GPU
```

**GPU Architecture List:**
- `7.5` - Tesla T4, RTX 2080
- `8.0` - A100
- `8.6` - RTX 3090, A6000, A40
- `8.9` - RTX 4090, L4, L40

#### Install with C++ CUDA Extensions

```bash
cd /path/to/nnDetection
FORCE_CUDA=1 pip install -e .
```

**What happens during build:**
- Compiles C++/CUDA extensions (NMS, box operations)
- Embeds RPATH for automatic PyTorch library discovery
- Creates `nndet._C` extension module
- Installs Python dependencies

#### Alternative: Using Build Script

For automated build with testing:

```bash
cd /path/to/nnDetection
./build_wheel.sh
```

This creates a wheel in `dist/` and tests it in an isolated environment.

### Verification

```bash
python -c "import torch; import nndet; import nndet._C; print('✓ Installation successful with C++ extensions!')"
```

---

## Code Changes Reference

### Files Modified (8 total)

#### 1. **nndet/csrc/cuda/nms.cu** - CUDA C++ Extensions

Updated deprecated PyTorch C++ APIs:
```cpp
// Before: AT_ASSERTM(tensor.type().is_cuda(), "must be CUDA tensor");
// After:
TORCH_CHECK(tensor.is_cuda(), "must be CUDA tensor");

// Before: AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.type(), ...
// After:
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), ...
```

#### 2. **nndet/utils/tensor.py** - Removed deprecated imports

```python
# Before: from torch._six import string_classes
# After:
string_classes = str  # Python 3+ compatibility
```

#### 3. **setup.py** - RPATH Fix for Runtime Library Discovery

Added automatic PyTorch library linking:
```python
# Set RPATH to find PyTorch libraries at runtime
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
extra_link_args = [
    f'-Wl,-rpath,{torch_lib_path}',           # Absolute path
    '-Wl,-rpath,$ORIGIN/../torch/lib',        # Relative path (portable)
    '-Wl,-rpath,$ORIGIN/../../torch/lib',     # Alternative layout
]
```

**Impact:** CUDA extensions now find PyTorch libraries automatically without manual `LD_LIBRARY_PATH` setup.

#### 4. **nndet/ptmodule/base_module.py** - Lightning 2.x API

```python
# Before:
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary

def validation_epoch_end(self, validation_step_outputs):
    # Process outputs...

# After:
import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import ModelSummary

def on_validation_epoch_end(self):
    # Process self.validation_step_outputs...
```

#### 5. **nndet/training/swa.py** - Lightning Callbacks

```python
# Before:
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities import rank_zero_warn

# After:
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.utilities import rank_zero_warn
```

#### 6. **nndet/io/datamodule/base.py** - DataModule Updates

```python
# Before:
import pytorch_lightning as pl

# After:
import lightning.pytorch as pl
```

#### 7. **nndet/ptmodule/retinaunet/base.py** - Output Caching

Lightning 2.x no longer passes outputs to `*_epoch_end` hooks:
```python
def __init__(self, ...):
    super().__init__(...)
    # Cache outputs manually
    self.training_step_outputs = []
    self.validation_step_outputs = []

def training_step(self, batch, batch_idx):
    output = {...}
    self.training_step_outputs.append(output)
    return output

def on_train_epoch_end(self):
    # Process self.training_step_outputs
    # ...
    self.training_step_outputs.clear()
```

#### 8. **pyproject.toml** - Dependencies

```toml
[project]
requires-python = ">=3.9,<3.13"
dependencies = [
    "lightning>=2.0.0",        # Was: pytorch_lightning>=1.3.1,<=1.4.2
    "torchmetrics>=0.11.0",    # Was: >=0.7.0,<=0.7.3
    "SimpleITK>=2.1.0",        # Was: <2.1.0
    # ... other dependencies
]

[build-system]
requires = [
    "setuptools>=67.0",        # Was: >=45
    "torch>=2.0.0",            # Was: >=1.10.0
]
```

---

## Testing

### Run Complete Test Suite

```bash
cd /path/to/nnDetection
./run_tests.sh
```

### Test Categories

| File | Tests | Purpose |
|------|-------|---------|
| `test_gpu_functionality.py` | 12 | CUDA extensions, NMS on GPU, box operations |
| `test_core_imports.py` | 12 | All module imports and dependencies |
| `test_architecture.py` | 8 | Model components and building blocks |
| `test_evaluators.py` | 7 | Detection & segmentation evaluators |
| `test_losses_matchers.py` | 5 | Loss functions, matchers, ensemblers |
| `test_data_pipeline.py` | 7 | Data loading, augmentation, transforms |
| `test_scripts.py` | 9 | Console scripts and planning modules |
| `test_utilities.py` | 3 | File IO and preprocessing utilities |

Or manually:
```bash
pytest tests/test_fast_suite.py tests/test_imports.py -v
```

---

## Troubleshooting

### Build Issues

#### `nvcc not found`

**Error:** `error: nvcc not found at '/usr/local/cuda/bin/nvcc'`

**Solution:**
```bash
# Install CUDA toolkit (see CUDA Toolkit Setup above)
sudo apt-get install -y cuda-toolkit-12-4

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
```

#### CUDA version mismatch

**Error:** Build succeeds but runtime fails with CUDA errors

**Solution:**
```bash
# Verify PyTorch CUDA version matches toolkit
python -c "import torch; print(torch.version.cuda)"  # Should show 12.4

# Verify CUDA toolkit version
nvcc --version  # Should show release 12.4

# If mismatch, reinstall PyTorch with correct CUDA version
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

The CUDA toolkit version must match PyTorch's CUDA version:
- PyTorch 2.6.0+cu124 → requires CUDA 12.4 toolkit
- PyTorch 2.9.0+cu128 → requires CUDA 12.8 toolkit

### Runtime Issues

#### `ImportError: libc10.so: cannot open shared object file`

**Error:** CUDA extensions fail to load

**Solution:** This should be automatically fixed by the RPATH settings in setup.py. If you still see this:

```bash
# Check if extension was built correctly
python -c "import nndet._C"

# If it fails, verify PyTorch installation
python -c "import torch; print(torch.__file__)"

# Rebuild with FORCE_CUDA
cd /path/to/nnDetection
pip uninstall nndet
FORCE_CUDA=1 pip install -e .
```

Or manually add to `~/.bashrc`:
```bash
# Add PyTorch libraries to LD_LIBRARY_PATH
if python -c "import torch" 2>/dev/null; then
    export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
fi
```

#### `ModuleNotFoundError: No module named 'nndet._C'`

**Error:** C++ extensions not built

**Solution:**
```bash
# Ensure CUDA is forced during build
FORCE_CUDA=1 pip install -e .

# Verify extensions exist
find . -name "_C*.so"
# Should show: nndet/_C.cpython-*.so
```

#### Symbol errors or undefined symbols

**Solution:** Clean rebuild:
```bash
cd /path/to/nnDetection
rm -rf build nndet.egg-info nndet/_C*.so
pip uninstall nndet -y

# Then follow installation steps above
FORCE_CUDA=1 pip install -e .
```