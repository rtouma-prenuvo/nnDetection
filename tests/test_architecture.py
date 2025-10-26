"""
Test suite for nnDetection architecture components
Tests model building blocks and instantiation
"""
import pytest
import torch


@pytest.mark.cuda
class TestArchitectureImports:
    """Test architecture components can be imported"""
    
    def test_conv_blocks_import(self):
        """Test convolution blocks module"""
        from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu
        from nndet.arch.blocks import basic, res
        
        assert ConvInstanceRelu is not None
        assert ConvGroupRelu is not None
        assert basic is not None
        assert res is not None
    
    def test_encoder_decoder_import(self):
        """Test encoder and decoder modules"""
        from nndet.arch.encoder import Encoder
        from nndet.arch.decoder.base import UFPNModular
        
        assert Encoder is not None
        assert UFPNModular is not None
    
    def test_heads_import(self):
        """Test detection heads"""
        from nndet.arch.heads import classifier, regressor, segmenter, comb
        
        assert classifier is not None
        assert regressor is not None
        assert segmenter is not None
        assert comb is not None


class TestModelRegistry:
    """Test model registry and module imports"""
    
    def test_module_registry(self):
        """Test module registry exists"""
        from nndet.ptmodule import MODULE_REGISTRY
        assert MODULE_REGISTRY is not None
    
    def test_retinaunet_modules(self):
        """Test RetinaUNet modules can be imported"""
        from nndet.ptmodule.retinaunet.base import RetinaUNetModule
        from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
        from nndet.ptmodule.base_module import LightningBaseModule
        
        assert RetinaUNetModule is not None
        assert RetinaUNetV001 is not None
        assert LightningBaseModule is not None


class TestAnchors:
    """Test anchor generation"""
    
    def test_anchor_module(self):
        """Test anchor module imports"""
        from nndet.core.boxes import anchors
        
        assert anchors is not None
