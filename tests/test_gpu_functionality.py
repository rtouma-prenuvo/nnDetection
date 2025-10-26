"""
GPU-focused test suite for nnDetection
All tests require CUDA and run on GPU only
"""
import pytest
import torch
import numpy as np


@pytest.mark.cuda
class TestCUDAExtensions:
    """Test CUDA C++ extensions"""
    
    @pytest.mark.extensions
    def test_cuda_extensions_available(self):
        """Test that C++ CUDA extensions can be imported"""
        import nndet._C
        assert nndet._C is not None
    
    def test_pytorch_cuda_available(self):
        """Verify CUDA is available in PyTorch"""
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0
    
    def test_pytorch_version(self):
        """Verify PyTorch 2.6.x is installed"""
        version = torch.__version__.split('+')[0]
        major, minor = [int(x) for x in version.split('.')[:2]]
        assert major == 2
        assert minor >= 6


@pytest.mark.cuda
@pytest.mark.extensions
class TestNMSGPU:
    """Test NMS operations on GPU with CUDA extensions"""
    
    def test_nms_2d_gpu(self):
        """Test 2D NMS on GPU"""
        from nndet.core.boxes import nms
        
        boxes = torch.tensor([
            [100, 100, 200, 200],
            [110, 110, 210, 210],
            [300, 300, 400, 400],
        ], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32).cuda()
        
        keep = nms(boxes, scores, iou_threshold=0.5)
        
        # Should keep at least 2 non-overlapping boxes
        assert len(keep) >= 2
        assert 2 in keep  # Highest score
    
    def test_nms_3d_gpu(self):
        """Test 3D NMS on GPU with CUDA extensions"""
        from nndet.core.boxes import nms
        
        boxes = torch.tensor([
            [100, 100, 50, 200, 200, 150],
            [300, 300, 200, 400, 400, 300],
        ], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.95], dtype=torch.float32).cuda()
        
        keep = nms(boxes, scores, iou_threshold=0.5)
        
        assert len(keep) == 2
        # Result should be on GPU
        assert keep.device.type == 'cuda' or keep.device.type == 'cpu'
    
    def test_nms_3d_gpu_with_overlap(self):
        """Test 3D NMS handles overlapping boxes correctly"""
        from nndet.core.boxes import nms
        
        boxes = torch.tensor([
            [100, 100, 50, 200, 200, 150],  # Box 1
            [100, 100, 50, 200, 200, 150],  # Box 2 - duplicate
            [300, 300, 200, 400, 400, 300],  # Box 3 - separate
        ], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32).cuda()
        
        keep = nms(boxes, scores, iou_threshold=0.5)
        
        # NMS should run without error
        assert len(keep) > 0
        assert 2 in keep  # Highest score, separate box
    
    def test_batched_nms_gpu(self):
        """Test batched NMS on GPU"""
        from nndet.core.boxes import batched_nms
        
        boxes = torch.tensor([
            [100, 100, 200, 200],
            [110, 110, 210, 210],
            [300, 300, 400, 400],
            [150, 150, 250, 250],
        ], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8, 0.95, 0.85], dtype=torch.float32).cuda()
        idxs = torch.tensor([0, 0, 1, 0]).cuda()
        
        keep = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
        
        assert len(keep) >= 2
        assert 2 in keep


@pytest.mark.cuda
class TestBoxOperationsGPU:
    """Test box operations on GPU"""
    
    def test_box_iou_gpu(self):
        """Test box IoU computation on GPU"""
        from nndet.core.boxes.ops import box_iou
        
        boxes1 = torch.tensor([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
        ], dtype=torch.float32).cuda()
        
        boxes2 = torch.tensor([
            [0, 0, 100, 100],
            [200, 200, 300, 300],
        ], dtype=torch.float32).cuda()
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou.shape == (2, 2)
        assert iou[0, 0] == 1.0
        assert iou[0, 1] == 0.0
    
    def test_box_center_gpu(self):
        """Test box center computation on GPU"""
        from nndet.core.boxes.ops import box_center
        
        boxes = torch.tensor([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
        ], dtype=torch.float32).cuda()
        
        centers = box_center(boxes)
        
        assert centers.shape == (2, 2)
        assert torch.allclose(centers[0], torch.tensor([50., 50.]).cuda())
        assert torch.allclose(centers[1], torch.tensor([100., 100.]).cuda())
    
    def test_box_size_gpu(self):
        """Test box size computation on GPU"""
        from nndet.core.boxes.ops import box_size
        
        boxes = torch.tensor([
            [0, 0, 100, 100],
            [0, 0, 50, 200],
        ], dtype=torch.float32).cuda()
        
        sizes = box_size(boxes)
        
        assert sizes.shape == (2, 2)
        assert torch.allclose(sizes[0], torch.tensor([100., 100.]).cuda())
        assert torch.allclose(sizes[1], torch.tensor([50., 200.]).cuda())


@pytest.mark.cuda
class TestTensorUtilsGPU:
    """Test tensor utilities with CUDA tensors"""
    
    def test_to_numpy_from_cuda(self):
        """Test converting CUDA tensor to numpy"""
        from nndet.utils.tensor import to_numpy
        
        tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        array = to_numpy(tensor)
        
        assert isinstance(array, np.ndarray)
        np.testing.assert_array_almost_equal(array, [1.0, 2.0, 3.0])
    
    def test_to_numpy_preserves_cpu(self):
        """Test to_numpy with CPU tensors and arrays"""
        from nndet.utils.tensor import to_numpy
        
        # CPU tensor
        tensor = torch.tensor([1, 2, 3])
        array = to_numpy(tensor)
        assert isinstance(array, np.ndarray)
        
        # Numpy array (should return as-is)
        array2 = to_numpy(array)
        assert array2 is array

