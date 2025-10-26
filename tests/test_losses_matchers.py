"""
Test suite for losses, matchers, and ensemblers
"""
import pytest
import torch


class TestLosses:
    """Test loss functions"""
    
    def test_loss_modules(self):
        """Test loss modules can be imported"""
        from nndet.losses import classification, regression, segmentation
        
        assert classification is not None
        assert regression is not None
        assert segmentation is not None
    
    @pytest.mark.cuda
    def test_focal_loss(self):
        """Test focal loss works on GPU"""
        from nndet.losses.classification import FocalLossWithLogits
        
        loss_fn = FocalLossWithLogits(alpha=0.25, gamma=2.0)
        
        pred = torch.randn(4, 3).cuda()
        target = torch.randint(0, 3, (4,)).cuda()
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert loss.device.type == 'cuda'


class TestMatchers:
    """Test box matching strategies"""
    
    def test_matcher_imports(self):
        """Test matcher modules"""
        from nndet.core.boxes.matcher import iou, atss, base
        
        assert iou is not None
        assert atss is not None
        assert base is not None


class TestEnsemblers:
    """Test ensembler functionality"""
    
    def test_ensembler_imports(self):
        """Test ensembler modules"""
        from nndet.inference.ensembler import detection, segmentation, base
        
        assert detection is not None
        assert segmentation is not None
        assert base is not None


class TestDetectionPostprocessing:
    """Test detection postprocessing"""
    
    def test_postprocessing_module(self):
        """Test postprocessing module"""
        from nndet.inference.detection import postprocessing, model
        
        assert postprocessing is not None
        assert model is not None
