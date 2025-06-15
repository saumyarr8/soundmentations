"""
Comprehensive test suite for padding transforms.
Tests all padding classes for mono audio support only.
"""
import pytest
import numpy as np
from unittest.mock import patch

from soundmentations.transforms.time.pad import (
    BasePad, Pad, CenterPad, StartPad, PadToLength, CenterPadToLength, PadToMultiple
)


class TestBasePad:
    """Test cases for the BasePad base class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        # Test default probability
        base_pad = BasePad(pad_length=1000)
        assert base_pad.pad_length == 1000
        assert base_pad.p == 1.0
        
        # Test custom parameters
        base_pad = BasePad(pad_length=2048, p=0.7)
        assert base_pad.pad_length == 2048
        assert base_pad.p == 0.7
    
    def test_init_invalid_pad_length_type(self):
        """Test initialization with invalid pad_length type."""
        with pytest.raises(TypeError, match="pad_length must be an integer"):
            BasePad(pad_length=1000.5)
        
        with pytest.raises(TypeError, match="pad_length must be an integer"):
            BasePad(pad_length="1000")
    
    def test_init_invalid_pad_length_value(self):
        """Test initialization with invalid pad_length value."""
        with pytest.raises(ValueError, match="pad_length must be positive"):
            BasePad(pad_length=0)
        
        with pytest.raises(ValueError, match="pad_length must be positive"):
            BasePad(pad_length=-100)
    
    def test_init_invalid_probability_type(self):
        """Test initialization with invalid probability types."""
        with pytest.raises(TypeError, match="p must be a float or an integer"):
            BasePad(pad_length=1000, p="0.5")
        
        with pytest.raises(TypeError, match="p must be a float or an integer"):
            BasePad(pad_length=1000, p=None)
    
    def test_init_invalid_probability_value(self):
        """Test initialization with invalid probability values."""
        with pytest.raises(ValueError, match="p must be between 0.0 and 1.0"):
            BasePad(pad_length=1000, p=-0.1)
        
        with pytest.raises(ValueError, match="p must be between 0.0 and 1.0"):
            BasePad(pad_length=1000, p=1.1)
    
    def test_call_invalid_sample_type(self):
        """Test __call__ with invalid sample type."""
        base_pad = BasePad(pad_length=1000)
        
        with pytest.raises(TypeError, match="sample must be a numpy array"):
            base_pad([1, 2, 3])
        
        with pytest.raises(TypeError, match="sample must be a numpy array"):
            base_pad("audio")
    
    def test_call_empty_sample(self):
        """Test __call__ with empty sample."""
        base_pad = BasePad(pad_length=1000)
        
        with pytest.raises(ValueError, match="sample cannot be empty"):
            base_pad(np.array([]))
    
    def test_call_invalid_sample_dimensions(self):
        """Test __call__ with invalid sample dimensions (mono audio only)."""
        base_pad = BasePad(pad_length=1000)
        
        # 2D arrays should be rejected since we only support mono audio
        with pytest.raises(ValueError, match="sample must be a 1D array \\(mono audio only\\)"):
            base_pad(np.array([[1, 2], [3, 4]]))
        
        # 3D arrays should be rejected
        with pytest.raises(ValueError, match="sample must be a 1D array \\(mono audio only\\)"):
            base_pad(np.array([[[1, 2]]]))
    
    @patch('random.random')
    def test_probability_skip(self, mock_random):
        """Test that transformation is skipped based on probability."""
        mock_random.return_value = 0.8  # Greater than p=0.5
        
        base_pad = BasePad(pad_length=1000, p=0.5)
        sample = np.array([1, 2, 3, 4, 5])
        
        result = base_pad(sample)
        np.testing.assert_array_equal(result, sample)
    
    def test_not_implemented_error(self):
        """Test that _pad raises NotImplementedError."""
        base_pad = BasePad(pad_length=1000)
        sample = np.array([1, 2, 3])
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement the _pad method"):
            base_pad._pad(sample)


class TestPad:
    """Test cases for the Pad class (end padding)."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        pad = Pad(pad_length=1000)
        assert pad.pad_length == 1000
        assert pad.p == 1.0
        
        pad = Pad(pad_length=2048, p=0.8)
        assert pad.pad_length == 2048
        assert pad.p == 0.8
    
    def test_pad_shorter_sample(self):
        """Test padding when sample is shorter than target length."""
        pad = Pad(pad_length=10)
        sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = pad(sample)
        expected = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 10
        assert result.dtype == np.float32
    
    def test_pad_longer_sample(self):
        """Test behavior when sample is longer than target length."""
        pad = Pad(pad_length=3)
        sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = pad(sample)
        # Should return unchanged since already longer than target
        np.testing.assert_array_equal(result, sample)
    
    def test_pad_equal_length_sample(self):
        """Test behavior when sample is exactly target length."""
        pad = Pad(pad_length=5)
        sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = pad(sample)
        # Should return unchanged since already at target length
        np.testing.assert_array_equal(result, sample)
    
    def test_preserve_dtype(self):
        """Test that padding preserves the original dtype."""
        pad = Pad(pad_length=8)
        
        for dtype in [np.float32, np.float64, np.int16, np.int32]:
            sample = np.array([1, 2, 3], dtype=dtype)
            result = pad(sample)
            
            assert result.dtype == dtype
            assert len(result) == 8
            # Check that padding is zeros of the same dtype
            np.testing.assert_array_equal(result[:3], sample)
            np.testing.assert_array_equal(result[3:], np.zeros(5, dtype=dtype))


class TestCenterPad:
    """Test cases for the CenterPad class (symmetric padding)."""
    
    def test_pad_even_padding(self):
        """Test center padding with even total padding needed."""
        pad = CenterPad(pad_length=10)
        sample = np.array([1, 2, 3, 4], dtype=np.float32)  # Need 6 zeros total
        
        result = pad(sample)
        expected = np.array([0, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 10
    
    def test_pad_odd_padding(self):
        """Test center padding with odd total padding needed."""
        pad = CenterPad(pad_length=8)
        sample = np.array([1, 2, 3], dtype=np.float32)  # Need 5 zeros total
        
        result = pad(sample)
        expected = np.array([0, 0, 1, 2, 3, 0, 0, 0], dtype=np.float32)  # 2 left, 3 right
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 8
    
    def test_pad_single_sample(self):
        """Test center padding with single sample."""
        pad = CenterPad(pad_length=5)
        sample = np.array([42], dtype=np.float32)  # Need 4 zeros total
        
        result = pad(sample)
        expected = np.array([0, 0, 42, 0, 0], dtype=np.float32)  # 2 left, 2 right
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 5
    
    def test_no_padding_needed(self):
        """Test when sample is already long enough."""
        pad = CenterPad(pad_length=3)
        sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = pad(sample)
        np.testing.assert_array_equal(result, sample)


class TestStartPad:
    """Test cases for the StartPad class (front padding)."""
    
    def test_pad_shorter_sample(self):
        """Test start padding when sample is shorter than target length."""
        pad = StartPad(pad_length=8)
        sample = np.array([1, 2, 3], dtype=np.float32)
        
        result = pad(sample)
        expected = np.array([0, 0, 0, 0, 0, 1, 2, 3], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 8
    
    def test_pad_longer_sample(self):
        """Test behavior when sample is longer than target length."""
        pad = StartPad(pad_length=2)
        sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = pad(sample)
        # Should return unchanged since already longer than target
        np.testing.assert_array_equal(result, sample)
    
    def test_pad_equal_length_sample(self):
        """Test behavior when sample is exactly target length."""
        pad = StartPad(pad_length=4)
        sample = np.array([1, 2, 3, 4], dtype=np.float32)
        
        result = pad(sample)
        # Should return unchanged since already at target length
        np.testing.assert_array_equal(result, sample)


class TestPadToLength:
    """Test cases for the PadToLength class (exact length with end operations)."""
    
    def test_pad_shorter_sample(self):
        """Test padding when sample is shorter than target length."""
        pad = PadToLength(pad_length=7)
        sample = np.array([1, 2, 3], dtype=np.float32)
        
        result = pad(sample)
        expected = np.array([1, 2, 3, 0, 0, 0, 0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 7
    
    def test_trim_longer_sample(self):
        """Test trimming when sample is longer than target length."""
        pad = PadToLength(pad_length=3)
        sample = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        
        result = pad(sample)
        expected = np.array([1, 2, 3], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 3
    
    def test_exact_length_sample(self):
        """Test behavior when sample is exactly target length."""
        pad = PadToLength(pad_length=4)
        sample = np.array([1, 2, 3, 4], dtype=np.float32)
        
        result = pad(sample)
        np.testing.assert_array_equal(result, sample)
        assert len(result) == 4


class TestCenterPadToLength:
    """Test cases for the CenterPadToLength class (exact length with center operations)."""
    
    def test_pad_shorter_sample(self):
        """Test center padding when sample is shorter than target length."""
        pad = CenterPadToLength(pad_length=9)
        sample = np.array([1, 2, 3], dtype=np.float32)  # Need 6 zeros total
        
        result = pad(sample)
        expected = np.array([0, 0, 0, 1, 2, 3, 0, 0, 0], dtype=np.float32)  # 3 left, 3 right
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 9
    
    def test_trim_longer_sample_even(self):
        """Test center trimming when sample is longer (even excess)."""
        pad = CenterPadToLength(pad_length=5)
        sample = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)  # Remove 4 total
        
        result = pad(sample)
        expected = np.array([3, 4, 5, 6, 7], dtype=np.float32)  # Remove 2 from each end
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 5
    
    def test_trim_longer_sample_odd(self):
        """Test center trimming when sample is longer (odd excess)."""
        pad = CenterPadToLength(pad_length=4)
        sample = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)  # Remove 3 total
        
        result = pad(sample)
        expected = np.array([2, 3, 4, 5], dtype=np.float32)  # Remove 1 from start, 2 from end
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 4
    
    def test_exact_length_sample(self):
        """Test behavior when sample is exactly target length."""
        pad = CenterPadToLength(pad_length=6)
        sample = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        
        result = pad(sample)
        np.testing.assert_array_equal(result, sample)
        assert len(result) == 6


class TestPadToMultiple:
    """Test cases for the PadToMultiple class (pad to multiple of value)."""
    
    def test_pad_to_multiple_needed(self):
        """Test padding when length is not a multiple of target."""
        pad = PadToMultiple(pad_length=4)
        sample = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)  # Length 6, need 2 more for 8
        
        result = pad(sample)
        expected = np.array([1, 2, 3, 4, 5, 6, 0, 0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 8
        assert len(result) % 4 == 0
    
    def test_no_padding_needed(self):
        """Test when length is already a multiple of target."""
        pad = PadToMultiple(pad_length=3)
        sample = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)  # Length 6, already multiple of 3
        
        result = pad(sample)
        np.testing.assert_array_equal(result, sample)
        assert len(result) == 6
        assert len(result) % 3 == 0
    
    def test_single_sample_padding(self):
        """Test padding with single sample."""
        pad = PadToMultiple(pad_length=5)
        sample = np.array([42], dtype=np.float32)  # Length 1, need 4 more for 5
        
        result = pad(sample)
        expected = np.array([42, 0, 0, 0, 0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 5
        assert len(result) % 5 == 0
    
    def test_stft_friendly_padding(self):
        """Test padding for common STFT use case."""
        pad = PadToMultiple(pad_length=1024)
        sample = np.array(range(2500), dtype=np.float32)  # Length 2500
        
        result = pad(sample)
        
        # Should pad to next multiple of 1024, which is 3072
        assert len(result) == 3072
        assert len(result) % 1024 == 0
        # Original data should be preserved
        np.testing.assert_array_equal(result[:2500], sample)
        # Padding should be zeros
        np.testing.assert_array_equal(result[2500:], np.zeros(572, dtype=np.float32))


class TestIntegrationAndEdgeCases:
    """Integration tests and edge case testing for mono audio only."""
    
    def test_different_dtypes(self):
        """Test padding with different numpy dtypes."""
        for dtype in [np.float32, np.float64, np.int16, np.int32]:
            pad = CenterPad(pad_length=8)
            sample = np.array([1, 2, 3], dtype=dtype)
            
            result = pad(sample)
            
            assert result.dtype == dtype
            assert len(result) == 8
    
    def test_very_small_samples(self):
        """Test padding with very small samples."""
        pad = Pad(pad_length=1000)
        sample = np.array([1], dtype=np.float32)  # Single sample
        
        result = pad(sample)
        
        assert len(result) == 1000
        assert result[0] == 1
        np.testing.assert_array_equal(result[1:], np.zeros(999, dtype=np.float32))
    
    def test_large_padding_requirements(self):
        """Test padding with large padding requirements."""
        pad = PadToLength(pad_length=100000)
        sample = np.array([1, 2, 3], dtype=np.float32)
        
        result = pad(sample)
        
        assert len(result) == 100000
        np.testing.assert_array_equal(result[:3], sample)
        np.testing.assert_array_equal(result[3:], np.zeros(99997, dtype=np.float32))
    
    @patch('random.random')
    def test_probability_behavior_across_classes(self, mock_random):
        """Test probability behavior is consistent across all pad classes."""
        mock_random.return_value = 0.9  # Greater than p=0.5
        
        sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        pad_classes = [
            Pad(pad_length=10, p=0.5),
            CenterPad(pad_length=10, p=0.5),
            StartPad(pad_length=10, p=0.5),
            PadToLength(pad_length=10, p=0.5),
            CenterPadToLength(pad_length=10, p=0.5),
            PadToMultiple(pad_length=8, p=0.5)
        ]
        
        for pad in pad_classes:
            result = pad(sample)
            # All should return original sample due to probability
            np.testing.assert_array_equal(result, sample)
    
    def test_zero_padding_verification(self):
        """Test that all padding is actually zeros and not some other value."""
        pad = CenterPad(pad_length=20)
        sample = np.array([1, 2, 3], dtype=np.float32)
        
        result = pad(sample)
        
        # Check that padding regions are exactly zero
        left_padding = result[:8]  # Should be 8 zeros on left
        right_padding = result[11:]  # Should be 9 zeros on right
        
        np.testing.assert_array_equal(left_padding, np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(right_padding, np.zeros(9, dtype=np.float32))
        
        # Check that original data is preserved
        np.testing.assert_array_equal(result[8:11], sample)
    
    def test_edge_case_boundary_calculations(self):
        """Test boundary calculations for edge cases."""
        # Test with pad length of 1
        pad = PadToLength(pad_length=1)
        sample = np.array([1, 2, 3], dtype=np.float32)
        
        result = pad(sample)
        expected = np.array([1], dtype=np.float32)  # Should trim to length 1
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 1
    
    def test_consistent_output_lengths(self):
        """Test that output lengths are always correct for exact-length operations."""
        target_lengths = [1, 5, 10, 100, 1000]
        
        for target_length in target_lengths:
            sample = np.array([1, 2, 3], dtype=np.float32)
            
            # Test PadToLength
            pad_to_length = PadToLength(pad_length=target_length)
            result1 = pad_to_length(sample)
            assert len(result1) == target_length
            
            # Test CenterPadToLength
            center_pad_to_length = CenterPadToLength(pad_length=target_length)
            result2 = center_pad_to_length(sample)
            assert len(result2) == target_length
    
    def test_multiple_operations_consistency(self):
        """Test that multiple padding operations give consistent results."""
        # Test that applying the same operation twice gives the same result
        pad = Pad(pad_length=10)
        sample = np.array([1, 2, 3], dtype=np.float32)
        
        result1 = pad(sample)
        result2 = pad(result1)  # Should be unchanged since already padded
        
        np.testing.assert_array_equal(result1, result2)
        assert len(result1) == len(result2) == 10
    
    def test_mono_audio_constraints(self):
        """Test that only mono audio (1D arrays) are accepted."""
        pad = Pad(pad_length=10)
        
        # Valid mono audio
        mono_sample = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = pad(mono_sample)
        assert result.shape == (10,)
        
        # Invalid: stereo audio should be rejected
        stereo_sample = np.random.randn(50, 2).astype(np.float32)
        with pytest.raises(ValueError, match="sample must be a 1D array \\(mono audio only\\)"):
            pad(stereo_sample)
        
        # Invalid: multichannel audio should be rejected
        multichannel_sample = np.random.randn(50, 5).astype(np.float32)
        with pytest.raises(ValueError, match="sample must be a 1D array \\(mono audio only\\)"):
            pad(multichannel_sample)
    
    def test_empty_and_minimal_samples(self):
        """Test behavior with empty and minimal samples."""
        pad = CenterPad(pad_length=5)
        
        # Empty sample should raise error
        with pytest.raises(ValueError, match="sample cannot be empty"):
            pad(np.array([]))
        
        # Single sample should work
        single_sample = np.array([42.0])
        result = pad(single_sample)
        assert len(result) == 5
        assert result[2] == 42.0  # Should be centered
    
    def test_numerical_precision(self):
        """Test numerical precision with different dtypes."""
        sample = np.array([1.5, 2.7, 3.9], dtype=np.float32)
        
        for PadClass in [Pad, CenterPad, StartPad, PadToLength, CenterPadToLength]:
            pad = PadClass(pad_length=8)
            result = pad(sample)
            
            # Check that original values are preserved exactly
            if PadClass == CenterPad:
                # For center pad, find where original data is
                start_idx = (8 - 3) // 2  # Should be 2
                np.testing.assert_array_equal(result[start_idx:start_idx+3], sample)
            elif PadClass == StartPad:
                # For start pad, original data is at the end
                np.testing.assert_array_equal(result[-3:], sample)
            elif PadClass == CenterPadToLength:
                # For center pad to length, find where original data is
                start_idx = (8 - 3) // 2  # Should be 2
                np.testing.assert_array_equal(result[start_idx:start_idx+3], sample)
            else:
                # For Pad and PadToLength, original data is at the beginning
                np.testing.assert_array_equal(result[:3], sample)
    
    def test_memory_efficiency(self):
        """Test that padding doesn't create unnecessary copies."""
        # Test with large sample to ensure memory efficiency
        large_sample = np.ones(10000, dtype=np.float32)
        
        # Test that shorter padding doesn't change the array
        pad = Pad(pad_length=5000)  # Shorter than sample
        result = pad(large_sample)
        
        # Should return the same array reference for efficiency
        assert result is large_sample
        
        # Test that longer padding creates new array
        pad_long = Pad(pad_length=15000)
        result_long = pad_long(large_sample)
        
        # Should be a new array
        assert result_long is not large_sample
        assert len(result_long) == 15000
