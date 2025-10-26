"""
Core imports test suite for nnDetection
Tests all critical module imports
"""
import pytest
from nndet.utils.info import SuppressPrint


class TestDependencyImports:
    """Test that all required dependencies are available"""
    
    def test_torch_torchvision(self):
        """Test PyTorch and torchvision"""
        import torch
        import torchvision
        
        assert torch is not None
        assert torchvision is not None
        
        # Verify versions
        version = torch.__version__.split('+')[0]
        major = int(version.split('.')[0])
        assert major >= 2, f"Expected PyTorch 2.x+, got {torch.__version__}"
    
    def test_lightning(self):
        """Test Lightning"""
        import lightning as L
        assert L is not None
    
    def test_nevergrad(self):
        """Test nevergrad"""
        import nevergrad as ng
        assert ng is not None
    
    def test_batchgenerators(self):
        """Test batchgenerators"""
        import batchgenerators
        assert batchgenerators is not None
    
    def test_nnunet(self):
        """Test nnU-Net"""
        with SuppressPrint():
            import nnunet.preprocessing.preprocessing as nn_preprocessing
            assert nn_preprocessing is not None


class TestCoreModules:
    """Test that core nnDetection modules can be imported"""
    
    def test_inference_modules(self):
        """Test inference module imports"""
        from nndet.inference.loading import load_all_models
        from nndet.inference.predictor import Predictor
        
        assert load_all_models is not None
        assert Predictor is not None
    
    def test_io_modules(self):
        """Test IO module imports"""
        from nndet.io.crop import crop_to_nonzero
        from nndet.io.load import load_pickle, save_pickle
        from nndet.io.paths import get_case_ids_from_dir
        
        assert crop_to_nonzero is not None
        assert load_pickle is not None
        assert save_pickle is not None
        assert get_case_ids_from_dir is not None
    
    def test_preprocessing_modules(self):
        """Test preprocessing module imports"""
        from nndet.preprocessing.preprocessor import GenericPreprocessor
        from nndet.preprocessing.resampling import resample_patient
        from nndet.preprocessing.crop import crop_to_bbox, create_nonzero_mask
        
        assert GenericPreprocessor is not None
        assert resample_patient is not None
        assert crop_to_bbox is not None
        assert create_nonzero_mask is not None
    
    def test_box_operations(self):
        """Test box operations imports"""
        from nndet.core.boxes import nms, batched_nms
        from nndet.core.boxes.ops import box_iou, box_center, box_size
        from nndet.core.boxes.coder import encode_boxes, decode_single, BoxCoderND
        
        assert nms is not None
        assert batched_nms is not None
        assert box_iou is not None
        assert box_center is not None
        assert box_size is not None
        assert encode_boxes is not None
        assert decode_single is not None
        assert BoxCoderND is not None
    
    def test_utilities(self):
        """Test utility module imports"""
        from nndet.utils.tensor import to_numpy
        from nndet.utils.config import load_dataset_info, compose
        from nndet.utils.check import env_guard
        
        assert to_numpy is not None
        assert load_dataset_info is not None
        assert compose is not None
        assert env_guard is not None
    
    def test_evaluator_modules(self):
        """Test evaluator module imports"""
        from nndet.evaluator.registry import save_metric_output
        
        assert save_metric_output is not None

