"""
Pytest configuration and fixtures for nnDetection tests
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_2d_boxes():
    """Fixture providing sample 2D bounding boxes"""
    return torch.tensor([
        [100, 100, 200, 200],
        [110, 110, 210, 210],  # Overlaps with first
        [300, 300, 400, 400],  # Separate box
    ], dtype=torch.float32)


@pytest.fixture
def sample_3d_boxes():
    """Fixture providing sample 3D bounding boxes"""
    return torch.tensor([
        [100, 100, 50, 200, 200, 150],
        [110, 110, 60, 210, 210, 160],  # Overlaps with first
        [300, 300, 200, 400, 400, 300],  # Separate box
    ], dtype=torch.float32)


@pytest.fixture
def sample_scores():
    """Fixture providing sample confidence scores"""
    return torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32)


@pytest.fixture
def sample_volume():
    """Fixture providing a sample 3D volume"""
    volume = np.zeros((100, 100, 100), dtype=np.float32)
    volume[20:80, 30:70, 10:90] = 1.0
    return volume


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "extensions: mark test as requiring C++ extensions"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on system capabilities"""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_extensions = pytest.mark.skip(reason="C++ extensions not available")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    # Check if C++ extensions are available
    extensions_available = False
    try:
        import nndet._C
        extensions_available = True
    except ImportError:
        pass
    
    for item in items:
        if "cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)
        if "extensions" in item.keywords and not extensions_available:
            item.add_marker(skip_extensions)

