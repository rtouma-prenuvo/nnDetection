"""
Test suite for data loading and augmentation pipeline
"""
import pytest
import torch
import numpy as np


class TestDataLoading:
    """Test data loading components"""
    
    def test_datamodule_module(self):
        """Test datamodule module exists"""
        from nndet.io.datamodule import bg_module
        
        assert bg_module is not None
    
    def test_patching_module(self):
        """Test patching module exists"""
        from nndet.io import patching
        
        assert patching is not None


class TestAugmentation:
    """Test augmentation components"""
    
    def test_augmentation_module(self):
        """Test augmentation modules"""
        from nndet.io.augmentation import base, bg_aug
        
        assert base is not None
        assert bg_aug is not None
    
    def test_transform_modules(self):
        """Test transform modules"""
        from nndet.io.transforms import base, spatial, instances
        
        assert base is not None
        assert spatial is not None
        assert instances is not None


class TestPaths:
    """Test path utilities"""
    
    def test_path_module(self):
        """Test path module and its functions"""
        from nndet.io import paths
        
        assert paths is not None
        assert hasattr(paths, 'get_case_ids_from_dir')
        assert hasattr(paths, 'get_case_id_from_path')


class TestITKLoading:
    """Test ITK/SimpleITK loading"""
    
    def test_itk_module(self):
        """Test ITK module"""
        from nndet.io import itk
        
        assert itk is not None
        assert hasattr(itk, 'load_sitk')
