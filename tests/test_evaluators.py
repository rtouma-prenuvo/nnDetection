"""
Test suite for nnDetection evaluators
Tests evaluation metrics and evaluator functionality
"""
import pytest
import numpy as np


class TestDetectionEvaluators:
    """Test detection evaluator functionality"""
    
    def test_box_evaluator_creation(self):
        """Test BoxEvaluator can be created"""
        from nndet.evaluator.det import BoxEvaluator
        
        evaluator = BoxEvaluator.create(
            classes=["class0", "class1"],
            fast=True,
        )
        
        assert evaluator is not None
        assert len(evaluator.metrics) > 0
    
    def test_detection_evaluator_import(self):
        """Test DetectionEvaluator can be imported"""
        from nndet.evaluator.det import DetectionEvaluator
        assert DetectionEvaluator is not None
    
    def test_metrics_imports(self):
        """Test metric modules"""
        from nndet.evaluator.detection import coco, froc
        
        assert coco is not None
        assert froc is not None


class TestSegmentationEvaluators:
    """Test segmentation evaluator functionality"""
    
    def test_segmentation_evaluator_creation(self):
        """Test SegmentationEvaluator can be created"""
        from nndet.evaluator.seg import SegmentationEvaluator
        
        evaluator = SegmentationEvaluator.create(per_class=False)
        
        assert evaluator is not None
    
    def test_segmentation_evaluator_import(self):
        """Test SegmentationEvaluator classes"""
        from nndet.evaluator.seg import SegmentationEvaluator, PerCaseSegmentationEvaluator
        
        assert SegmentationEvaluator is not None
        assert PerCaseSegmentationEvaluator is not None


class TestMatching:
    """Test detection matching functionality"""
    
    def test_matching_module(self):
        """Test matching module exists"""
        from nndet.evaluator.detection import matching
        assert matching is not None


class TestEvaluatorRegistry:
    """Test evaluator registry functions"""
    
    def test_registry_module(self):
        """Test evaluator registry module"""
        from nndet.evaluator import registry
        
        assert registry is not None
        assert hasattr(registry, 'save_metric_output')
