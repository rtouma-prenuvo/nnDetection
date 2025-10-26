"""
Test suite for nnDetection utilities
Lightweight tests for IO and preprocessing utilities
"""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path


class TestFileIO:
    """Test file loading operations"""
    
    def test_load_save_pickle(self):
        """Test pickle loading and saving"""
        from nndet.io.load import load_pickle, save_pickle
        
        test_data = {
            'boxes': np.array([[1, 2, 3, 4]]),
            'scores': np.array([0.9]),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            
            save_pickle(test_data, filepath)
            assert filepath.exists()
            
            loaded_data = load_pickle(filepath)
            
            np.testing.assert_array_equal(loaded_data['boxes'], test_data['boxes'])
            np.testing.assert_array_equal(loaded_data['scores'], test_data['scores'])


class TestPreprocessorImport:
    """Test preprocessor can be imported (not instantiated - too heavy)"""
    
    def test_preprocessor_import(self):
        """Test that GenericPreprocessor can be imported"""
        from nndet.preprocessing.preprocessor import GenericPreprocessor
        assert GenericPreprocessor is not None
        assert callable(GenericPreprocessor)


class TestCropOperations:
    """Test lightweight crop operations"""
    
    def test_crop_to_nonzero_small_volume(self):
        """Test crop to nonzero with small volume"""
        from nndet.io.crop import crop_to_nonzero
        
        # Small volume to keep test fast
        volume = np.zeros((20, 20, 20), dtype=np.float32)
        volume[5:15, 5:15, 5:15] = 1.0
        
        cropped_data, cropped_seg, bbox = crop_to_nonzero(volume)
        
        # Should have removed empty space
        assert cropped_data.shape[0] <= volume.shape[0]
        assert np.any(cropped_data > 0)
        assert bbox is not None

