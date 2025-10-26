"""
Test suite for nnDetection scripts
Tests that all console scripts can be imported
"""
import pytest


class TestScripts:
    """Test all console scripts can be imported"""
    
    def test_generate_example_script(self):
        """Test generate_example script"""
        from scripts.generate_example import main
        assert callable(main)
    
    def test_preprocess_script(self):
        """Test preprocess script"""
        from scripts.preprocess import main
        assert callable(main)
    
    def test_convert_scripts(self):
        """Test conversion scripts"""
        from scripts.convert_cls2fg import main as cls2fg_main
        from scripts.convert_seg2det import main as seg2det_main
        
        assert callable(cls2fg_main)
        assert callable(seg2det_main)
    
    def test_predict_script(self):
        """Test predict script"""
        from scripts.predict import main
        assert callable(main)
    
    def test_consolidate_script(self):
        """Test consolidate script"""
        from scripts.consolidate import main
        assert callable(main)
    
    def test_utils_scripts(self):
        """Test utility scripts"""
        from scripts.utils import (
            boxes2nii,
            seg2nii,
            unpack,
            env,
            hydra_searchpath
        )
        
        assert callable(boxes2nii)
        assert callable(seg2nii)
        assert callable(unpack)
        assert callable(env)
        assert callable(hydra_searchpath)


class TestPlanningModules:
    """Test planning and analysis modules"""
    
    def test_planning_modules(self):
        """Test planning modules exist"""
        from nndet.planning import analyzer, estimator
        from nndet.planning.architecture.boxes import base
        from nndet.planning.experiment import base as exp_base
        
        assert analyzer is not None
        assert estimator is not None
        assert base is not None
        assert exp_base is not None
