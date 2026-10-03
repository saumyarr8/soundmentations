"""
Unit tests for compressor transforms.
Tests compressor classes for mono audio support only.
"""
import pytest
import numpy as np
from unittest.mock import patch

from soundmentations.transforms.amplitude.compressor import BaseCompressor, Compressor


class TestBaseCompressor:
    """Test cases for the BaseCompressor base class."""
    
    def test_init_valid_probability(self):
        """Test initialization with valid probability values."""
        # Test default probability
        base_compressor = BaseCompressor()
        assert base_compressor.p == 1.0
        
        # Test custom probabilities
        for p in [0.0, 0.5, 1.0]:
            base_compressor = BaseCompressor(p=p)
            assert base_compressor.p == p
    
    def test_call_invalid_samples_type(self):
        """Test __call__ with invalid samples type."""
        base_compressor = BaseCompressor()
        
        with pytest.raises(TypeError):
            base_compressor([1, 2, 3], 44100)
        
        with pytest.raises(TypeError):
            base_compressor("audio", 44100)
    
    def test_call_empty_samples(self):
        """Test __call__ with empty samples."""
        base_compressor = BaseCompressor()
        
        with pytest.raises(ValueError):
            base_compressor(np.array([]), 44100)
    
    def test_call_non_1d_samples(self):
        """Test __call__ with non-1D samples."""
        base_compressor = BaseCompressor()
        
        with pytest.raises(ValueError):
            base_compressor(np.array([[1, 2], [3, 4]]), 44100)
    
    def test_call_invalid_sample_rate_type(self):
        """Test __call__ with invalid sample rate type."""
        base_compressor = BaseCompressor()
        samples = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(TypeError):
            base_compressor(samples, "44100")
        
        with pytest.raises(TypeError):
            base_compressor(samples, 44100.5)
    
    def test_call_invalid_sample_rate_value(self):
        """Test __call__ with invalid sample rate values."""
        base_compressor = BaseCompressor()
        samples = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            base_compressor(samples, 0)
        
        with pytest.raises(ValueError):
            base_compressor(samples, -44100)
    
    @patch('random.random')
    def test_probability_skip(self, mock_random):
        """Test that transformation is skipped based on probability."""
        mock_random.return_value = 0.8  # Greater than p=0.5
        
        base_compressor = BaseCompressor(p=0.5)
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = base_compressor(samples, 44100)
        np.testing.assert_array_equal(result, samples)
    
    def test_not_implemented_error(self):
        """Test that _compress raises NotImplementedError."""
        base_compressor = BaseCompressor()
        samples = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(NotImplementedError):
            base_compressor._compress(samples, 44100)


class TestCompressor:
    """Test cases for the Compressor class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        # Test default parameters
        compressor = Compressor()
        assert compressor.threshold == -12.0
        assert compressor.ratio == 4.0
        assert compressor.attack == 0.003
        assert compressor.release == 0.1
        assert compressor.p == 1.0
        
        # Test custom parameters
        compressor = Compressor(
            threshold=-6.0, 
            ratio=2.0, 
            attack=0.001, 
            release=0.05, 
            p=0.7
        )
        assert compressor.threshold == -6.0
        assert compressor.ratio == 2.0
        assert compressor.attack == 0.001
        assert compressor.release == 0.05
        assert compressor.p == 0.7
    
    def test_init_invalid_threshold_type(self):
        """Test initialization with invalid threshold type."""
        with pytest.raises(TypeError):
            Compressor(threshold="6.0")
        
        with pytest.raises(TypeError):
            Compressor(threshold=None)
    
    def test_init_invalid_ratio_type(self):
        """Test initialization with invalid ratio type."""
        with pytest.raises(TypeError):
            Compressor(ratio="4.0")
        
        with pytest.raises(TypeError):
            Compressor(ratio=None)
    
    def test_init_invalid_ratio_value(self):
        """Test initialization with invalid ratio value."""
        with pytest.raises(ValueError):
            Compressor(ratio=0.5)  # Must be >= 1.0
        
        with pytest.raises(ValueError):
            Compressor(ratio=0.0)
        
        with pytest.raises(ValueError):
            Compressor(ratio=-1.0)
    
    def test_init_invalid_attack_type(self):
        """Test initialization with invalid attack type."""
        with pytest.raises(TypeError):
            Compressor(attack="0.003")
        
        with pytest.raises(TypeError):
            Compressor(attack=None)
    
    def test_init_invalid_attack_value(self):
        """Test initialization with invalid attack value."""
        with pytest.raises(ValueError):
            Compressor(attack=0.0)
        
        with pytest.raises(ValueError):
            Compressor(attack=-0.001)
    
    def test_init_invalid_release_type(self):
        """Test initialization with invalid release type."""
        with pytest.raises(TypeError):
            Compressor(release="0.1")
        
        with pytest.raises(TypeError):
            Compressor(release=None)
    
    def test_init_invalid_release_value(self):
        """Test initialization with invalid release value."""
        with pytest.raises(ValueError):
            Compressor(release=0.0)
        
        with pytest.raises(ValueError):
            Compressor(release=-0.1)
    
    def test_basic_compression_functionality(self):
        """Test basic compression functionality."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        
        # Create samples that exceed the threshold
        samples = np.array([0.8, -0.8, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        result = compressor(samples, 44100)
        
        # Result should have same length
        assert len(result) == len(samples)
        
        # High amplitude samples should be compressed
        threshold_linear = 10 ** (compressor.threshold / 20)
        high_amp_indices = np.abs(samples) > threshold_linear
        
        if np.any(high_amp_indices):
            # Compressed samples should have lower amplitude than input
            # (allowing some tolerance for attack/release effects)
            high_amp_original = np.abs(samples[high_amp_indices])
            high_amp_compressed = np.abs(result[high_amp_indices])
            
            # Most high amplitude samples should be reduced
            assert np.mean(high_amp_compressed < high_amp_original * 0.9) > 0.5
    
    def test_no_compression_below_threshold(self):
        """Test that signals below threshold are not compressed."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        threshold_linear = 10 ** (compressor.threshold / 20)
        
        # Create samples below threshold
        samples = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        assert np.all(np.abs(samples) < threshold_linear)
        
        result = compressor(samples, 44100)
        
        # Samples should be relatively unchanged
        np.testing.assert_array_almost_equal(result, samples, decimal=2)
    
    def test_compression_ratios(self):
        """Test different compression ratios."""
        samples = np.array([1.0, -1.0], dtype=np.float32)  # High amplitude
        
        for ratio in [1.0, 2.0, 4.0, 8.0, 100.0]:  # Last one is essentially limiting
            compressor = Compressor(threshold=-6.0, ratio=ratio)
            result = compressor(samples, 44100)
            
            assert len(result) == len(samples)
            
            if ratio > 1.0:
                # Higher ratios should produce more compression
                assert np.all(np.abs(result) <= np.abs(samples))
    
    def test_attack_and_release_times(self):
        """Test compression with different attack and release times."""
        # Fast attack/release
        compressor_fast = Compressor(
            threshold=-6.0, 
            ratio=4.0, 
            attack=0.001, 
            release=0.01
        )
        
        # Slow attack/release
        compressor_slow = Compressor(
            threshold=-6.0, 
            ratio=4.0, 
            attack=0.01, 
            release=0.1
        )
        
        # Create impulse signal
        samples = np.zeros(44100, dtype=np.float32)
        samples[1000:1100] = 1.0  # Brief loud section
        
        result_fast = compressor_fast(samples, 44100)
        result_slow = compressor_slow(samples, 44100)
        
        assert len(result_fast) == len(samples)
        assert len(result_slow) == len(samples)
        
        # Both should reduce the loud section, but with different characteristics
        loud_section_fast = result_fast[1000:1100]
        loud_section_slow = result_slow[1000:1100]
        
        assert np.max(loud_section_fast) < 1.0
        assert np.max(loud_section_slow) < 1.0
    
    def test_different_sample_rates(self):
        """Test compressor works with different sample rates."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.array([1.0, -1.0, 0.8, -0.8], dtype=np.float32)
        
        for sample_rate in [22050, 44100, 48000, 96000]:
            result = compressor(samples, sample_rate)
            assert len(result) == len(samples)
            # Should compress high amplitude samples
            assert np.all(np.abs(result) <= np.abs(samples))
    
    @patch('random.random')
    def test_probability_behavior(self, mock_random):
        """Test probability behavior."""
        mock_random.return_value = 0.3  # Less than p=0.5
        
        compressor = Compressor(threshold=-6.0, ratio=4.0, p=0.5)
        samples = np.array([1.0, -1.0], dtype=np.float32)
        
        result = compressor(samples, 44100)
        # Should apply compression
        assert np.all(np.abs(result) <= np.abs(samples))
        
        mock_random.return_value = 0.7  # Greater than p=0.5
        result = compressor(samples, 44100)
        # Should not apply compression
        np.testing.assert_array_equal(result, samples)
    
    def test_preserves_shape_and_dtype(self):
        """Test that compressor preserves array shape and dtype."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.random.randn(1000).astype(np.float32)
        
        result = compressor(samples, 44100)
        assert result.shape == samples.shape
        assert result.dtype == samples.dtype
    
    def test_zero_samples(self):
        """Test compressor on zero samples."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.zeros(1000, dtype=np.float32)
        
        result = compressor(samples, 44100)
        np.testing.assert_array_equal(result, samples)
    
    def test_signs_preserved(self):
        """Test that compression preserves the sign of samples."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.array([0.8, -0.8, 0.5, -0.5], dtype=np.float32)
        
        result = compressor(samples, 44100)
        
        # Signs should be preserved
        assert np.all(np.sign(result) == np.sign(samples))


class TestIntegrationAndEdgeCases:
    """Integration tests and edge case testing for mono audio only."""
    
    def test_very_short_audio(self):
        """Test compression on very short audio samples."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.array([1.0, -1.0], dtype=np.float32)
        
        result = compressor(samples, 44100)
        assert len(result) == 2
        assert np.all(np.abs(result) <= np.abs(samples))
    
    def test_single_sample_audio(self):
        """Test compression on single sample."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.array([1.0], dtype=np.float32)
        
        result = compressor(samples, 44100)
        assert len(result) == 1
        assert np.abs(result[0]) <= np.abs(samples[0])
    
    def test_large_audio(self):
        """Test compression on large audio arrays."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.random.randn(1000000).astype(np.float32)
        
        result = compressor(samples, 44100)
        assert len(result) == len(samples)
        assert result.dtype == np.float32
    
    def test_extreme_threshold_values(self):
        """Test compressor with extreme threshold values."""
        samples = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        # Very low threshold (compress everything)
        compressor_low = Compressor(threshold=-40.0, ratio=4.0)
        result_low = compressor_low(samples, 44100)
        
        # Should compress all samples
        assert np.all(np.abs(result_low) <= np.abs(samples))
        
        # Very high threshold (compress nothing)
        compressor_high = Compressor(threshold=0.0, ratio=4.0)
        result_high = compressor_high(samples, 44100)
        
        # Should barely compress anything
        np.testing.assert_array_almost_equal(result_high, samples, decimal=2)
    
    def test_extreme_ratio_values(self):
        """Test compressor with extreme ratio values."""
        samples = np.array([1.0, -1.0], dtype=np.float32)
        
        # Ratio of 1.0 (no compression)
        compressor_no = Compressor(threshold=-6.0, ratio=1.0)
        result_no = compressor_no(samples, 44100)
        np.testing.assert_array_almost_equal(result_no, samples, decimal=2)
        
        # Very high ratio (limiting)
        compressor_limit = Compressor(threshold=-6.0, ratio=100.0)
        result_limit = compressor_limit(samples, 44100)
        
        threshold_linear = 10 ** (-6.0 / 20)
        # Should be heavily compressed/limited
        assert np.all(np.abs(result_limit) <= threshold_linear * 1.1)
    
    def test_alternating_values(self):
        """Test compression on alternating high/low values."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.array([1.0, 0.1, -1.0, 0.2, 0.9, -0.1], dtype=np.float32)
        
        result = compressor(samples, 44100)
        
        assert len(result) == len(samples)
        
        # High values should be compressed more than low values
        threshold_linear = 10 ** (compressor.threshold / 20)
        high_indices = np.abs(samples) > threshold_linear
        low_indices = np.abs(samples) <= threshold_linear
        
        if np.any(high_indices) and np.any(low_indices):
            # Check that high amplitude samples were compressed more
            high_compression_ratio = np.abs(result[high_indices]) / np.abs(samples[high_indices])
            low_compression_ratio = np.abs(result[low_indices]) / np.abs(samples[low_indices])
            
            assert np.mean(high_compression_ratio) < np.mean(low_compression_ratio)
    
    def test_deterministic_behavior(self):
        """Test that compression is deterministic."""
        compressor = Compressor(threshold=-6.0, ratio=4.0)
        samples = np.random.randn(1000).astype(np.float32)
        
        result1 = compressor(samples, 44100)
        result2 = compressor(samples, 44100)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_compression_effectiveness(self):
        """Test that compression actually reduces dynamic range."""
        compressor = Compressor(threshold=-12.0, ratio=4.0)
        
        # Create signal with wide dynamic range
        samples = np.concatenate([
            np.ones(1000) * 0.1,  # Quiet section
            np.ones(1000) * 1.0,  # Loud section
            np.ones(1000) * 0.05  # Very quiet section
        ]).astype(np.float32)
        
        result = compressor(samples, 44100)
        
        # Calculate dynamic range before and after
        original_range = np.max(np.abs(samples)) - np.min(np.abs(samples[samples != 0]))
        compressed_range = np.max(np.abs(result)) - np.min(np.abs(result[result != 0]))
        
        # Compressed signal should have smaller dynamic range
        assert compressed_range < original_range
