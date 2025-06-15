import pytest
import numpy as np
from unittest.mock import patch

from soundmentations.transforms.time.trim import (
    BaseTrim, Trim, RandomTrim, StartTrim, EndTrim, CenterTrim
)


class TestBaseTrim:
    """Test cases for the BaseTrim base class."""
    
    def test_init_valid_probability(self):
        """Test initialization with valid probability values."""
        # Test default probability
        base_trim = BaseTrim()
        assert base_trim.p == 1.0
        
        # Test custom probabilities
        for p in [0.0, 0.5, 1.0]:
            base_trim = BaseTrim(p=p)
            assert base_trim.p == p
    
    def test_init_invalid_probability_type(self):
        """Test initialization with invalid probability types."""
        with pytest.raises(TypeError, match="p must be a float or an integer"):
            BaseTrim(p="0.5")
        
        with pytest.raises(TypeError, match="p must be a float or an integer"):
            BaseTrim(p=None)
    
    def test_init_invalid_probability_value(self):
        """Test initialization with invalid probability values."""
        with pytest.raises(ValueError, match="p must be between 0.0 and 1.0"):
            BaseTrim(p=-0.1)
        
        with pytest.raises(ValueError, match="p must be between 0.0 and 1.0"):
            BaseTrim(p=1.1)
    
    def test_call_invalid_samples_type(self):
        """Test __call__ with invalid samples type."""
        base_trim = BaseTrim()
        
        with pytest.raises(TypeError, match="samples must be a numpy array"):
            base_trim([1, 2, 3], 44100)
        
        with pytest.raises(TypeError, match="samples must be a numpy array"):
            base_trim("audio", 44100)
    
    def test_call_empty_samples(self):
        """Test __call__ with empty samples."""
        base_trim = BaseTrim()
        
        with pytest.raises(ValueError, match="Input samples cannot be empty"):
            base_trim(np.array([]), 44100)
    
    def test_call_invalid_sample_rate_type(self):
        """Test __call__ with invalid sample rate type."""
        base_trim = BaseTrim()
        samples = np.array([1, 2, 3])
        
        with pytest.raises(TypeError, match="sample_rate must be an integer"):
            base_trim(samples, 44100.0)
        
        with pytest.raises(TypeError, match="sample_rate must be an integer"):
            base_trim(samples, "44100")
    
    def test_call_invalid_sample_rate_value(self):
        """Test __call__ with invalid sample rate values."""
        base_trim = BaseTrim()
        samples = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            base_trim(samples, 0)
        
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            base_trim(samples, -44100)
    
    @patch('random.random')
    def test_probability_skip(self, mock_random):
        """Test that transformation is skipped based on probability."""
        mock_random.return_value = 0.7  # Greater than p=0.5
        
        base_trim = BaseTrim(p=0.5)
        samples = np.array([1, 2, 3, 4, 5])
        
        result = base_trim(samples, 44100)
        np.testing.assert_array_equal(result, samples)
    
    def test_not_implemented_error(self):
        """Test that _trim raises NotImplementedError."""
        base_trim = BaseTrim()
        samples = np.array([1, 2, 3])
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement the _trim method"):
            base_trim._trim(samples, 44100)


class TestTrim:
    """Test cases for the Trim class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        # Test default values
        trim = Trim()
        assert trim.start_time == 0.0
        assert trim.end_time is None
        assert trim.p == 1.0
        
        # Test custom values
        trim = Trim(start_time=1.5, end_time=3.0, p=0.8)
        assert trim.start_time == 1.5
        assert trim.end_time == 3.0
        assert trim.p == 0.8
    
    def test_init_invalid_start_time_type(self):
        """Test initialization with invalid start_time type."""
        with pytest.raises(TypeError, match="start_time must be a number"):
            Trim(start_time="1.5")
    
    def test_init_negative_start_time(self):
        """Test initialization with negative start_time."""
        with pytest.raises(ValueError, match="start_time must be non-negative"):
            Trim(start_time=-1.0)
    
    def test_init_invalid_end_time_type(self):
        """Test initialization with invalid end_time type."""
        with pytest.raises(TypeError, match="end_time must be a number"):
            Trim(start_time=0.0, end_time="3.0")
    
    def test_init_invalid_end_time_value(self):
        """Test initialization with end_time <= start_time."""
        with pytest.raises(ValueError, match="end_time must be greater than start_time"):
            Trim(start_time=2.0, end_time=1.0)
        
        with pytest.raises(ValueError, match="end_time must be greater than start_time"):
            Trim(start_time=2.0, end_time=2.0)
    
    def test_trim_basic_functionality(self):
        """Test basic trimming functionality."""
        # Create test audio: 5 seconds at 44100 Hz
        sample_rate = 44100
        samples = np.arange(5 * sample_rate, dtype=np.float32)
        
        # Trim from 1s to 3s
        trim = Trim(start_time=1.0, end_time=3.0)
        result = trim(samples, sample_rate)
        
        expected_start = int(1.0 * sample_rate)
        expected_end = int(3.0 * sample_rate)
        expected = samples[expected_start:expected_end]
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 2 * sample_rate  # 2 seconds
    
    def test_trim_no_end_time(self):
        """Test trimming with no end_time (trim to end)."""
        sample_rate = 44100
        samples = np.arange(3 * sample_rate, dtype=np.float32)
        
        trim = Trim(start_time=1.0)
        result = trim(samples, sample_rate)
        
        expected_start = int(1.0 * sample_rate)
        expected = samples[expected_start:]
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 2 * sample_rate  # 2 seconds remaining
    
    def test_trim_start_time_exceeds_duration(self):
        """Test error when start_time exceeds audio duration."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)  # 2 seconds
        
        trim = Trim(start_time=3.0)  # 3 seconds > 2 seconds
        
        with pytest.raises(ValueError, match="start_time \\(3.0s\\) exceeds audio duration \\(2.00s\\)"):
            trim(samples, sample_rate)
    
    def test_trim_end_time_exceeds_duration(self):
        """Test error when end_time exceeds audio duration."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)  # 2 seconds
        
        trim = Trim(start_time=0.5, end_time=3.0)  # 3 seconds > 2 seconds
        
        with pytest.raises(ValueError, match="end_time \\(3.0s\\) exceeds audio duration \\(2.00s\\)"):
            trim(samples, sample_rate)
    
    def test_trim_no_audio_remains(self):
        """Test error when no audio remains after trimming."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)
        
        # This should not happen with valid initialization, but test edge case
        trim = Trim(start_time=1.0, end_time=2.0)
        # Manually set invalid values to test the check
        trim.start_time = 1.5
        trim.end_time = 1.0
        
        with pytest.raises(ValueError, match="No audio remains after trimming"):
            trim._trim(samples, sample_rate)


class TestRandomTrim:
    """Test cases for the RandomTrim class."""
    
    def test_init_single_duration(self):
        """Test initialization with single duration value."""
        trim = RandomTrim(duration=2.0)
        assert trim.min_duration == 2.0
        assert trim.max_duration == 2.0
        
        trim = RandomTrim(duration=5)  # integer
        assert trim.min_duration == 5
        assert trim.max_duration == 5
    
    def test_init_duration_range(self):
        """Test initialization with duration range."""
        trim = RandomTrim(duration=(1.0, 3.0))
        assert trim.min_duration == 1.0
        assert trim.max_duration == 3.0
        
        # Test with list
        trim = RandomTrim(duration=[1.5, 4.5])
        assert trim.min_duration == 1.5
        assert trim.max_duration == 4.5
    
    def test_init_invalid_duration_type(self):
        """Test initialization with invalid duration type."""
        with pytest.raises(ValueError, match="duration must be a float or tuple"):
            RandomTrim(duration="2.0")
        
        with pytest.raises(ValueError, match="duration must be a float or tuple"):
            RandomTrim(duration=(1.0, 2.0, 3.0))  # Too many values
    
    def test_init_negative_duration(self):
        """Test initialization with negative duration."""
        with pytest.raises(ValueError, match="duration must be positive"):
            RandomTrim(duration=-1.0)
        
        with pytest.raises(ValueError, match="duration values must be positive"):
            RandomTrim(duration=(-1.0, 2.0))
        
        with pytest.raises(ValueError, match="duration values must be positive"):
            RandomTrim(duration=(1.0, -2.0))
    
    def test_init_invalid_duration_range(self):
        """Test initialization with invalid duration range."""
        with pytest.raises(ValueError, match="min_duration must be <= max_duration"):
            RandomTrim(duration=(3.0, 1.0))
    
    def test_init_invalid_duration_values_type(self):
        """Test initialization with invalid types in duration tuple."""
        with pytest.raises(TypeError, match="duration values must be numbers"):
            RandomTrim(duration=("1.0", 2.0))
        
        with pytest.raises(TypeError, match="duration values must be numbers"):
            RandomTrim(duration=(1.0, "2.0"))
    
    @patch('random.uniform')
    def test_trim_fixed_duration(self, mock_uniform):
        """Test random trimming with fixed duration."""
        # Mock random.uniform to return predictable values
        mock_uniform.side_effect = [2.0, 1.0]  # target_duration, start_time
        
        sample_rate = 44100
        samples = np.arange(5 * sample_rate, dtype=np.float32)  # 5 seconds
        
        trim = RandomTrim(duration=2.0)
        result = trim(samples, sample_rate)
        
        # Expected: start at 1.0s, duration 2.0s
        expected_start = int(1.0 * sample_rate)
        expected_end = int(3.0 * sample_rate)
        expected = samples[expected_start:expected_end]
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 2 * sample_rate
    
    @patch('random.uniform')
    def test_trim_variable_duration(self, mock_uniform):
        """Test random trimming with variable duration."""
        # Mock random.uniform calls: target_duration=1.5, start_time=0.5
        mock_uniform.side_effect = [1.5, 0.5]
        
        sample_rate = 44100
        samples = np.arange(3 * sample_rate, dtype=np.float32)  # 3 seconds
        
        trim = RandomTrim(duration=(1.0, 2.0))
        result = trim(samples, sample_rate)
        
        # Expected: start at 0.5s, duration 1.5s
        expected_start = int(0.5 * sample_rate)
        expected_end = int(2.0 * sample_rate)
        expected = samples[expected_start:expected_end]
        
        np.testing.assert_array_equal(result, expected)
    
    def test_trim_duration_exceeds_audio(self):
        """Test error when target duration exceeds audio length."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)  # 2 seconds
        
        trim = RandomTrim(duration=3.0)  # 3 seconds > 2 seconds
        
        with pytest.raises(ValueError, match="target_duration \\(3.00s\\) exceeds audio duration \\(2.00s\\)"):
            trim(samples, sample_rate)


class TestStartTrim:
    """Test cases for the StartTrim class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        trim = StartTrim()
        assert trim.start_time == 0.0
        assert trim.p == 1.0
        
        trim = StartTrim(start_time=2.5, p=0.7)
        assert trim.start_time == 2.5
        assert trim.p == 0.7
    
    def test_init_invalid_start_time_type(self):
        """Test initialization with invalid start_time type."""
        with pytest.raises(TypeError, match="start_time must be a number"):
            StartTrim(start_time="2.0")
    
    def test_init_negative_start_time(self):
        """Test initialization with negative start_time."""
        with pytest.raises(ValueError, match="start_time must be non-negative"):
            StartTrim(start_time=-1.0)
    
    def test_trim_basic_functionality(self):
        """Test basic start trimming functionality."""
        sample_rate = 44100
        samples = np.arange(5 * sample_rate, dtype=np.float32)  # 5 seconds
        
        trim = StartTrim(start_time=2.0)
        result = trim(samples, sample_rate)
        
        expected_start = int(2.0 * sample_rate)
        expected = samples[expected_start:]
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 3 * sample_rate  # 3 seconds remaining
    
    def test_trim_start_time_exceeds_duration(self):
        """Test error when start_time exceeds audio duration."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)  # 2 seconds
        
        trim = StartTrim(start_time=3.0)  # 3 seconds > 2 seconds
        
        with pytest.raises(ValueError, match="start_time \\(3.0s\\) exceeds audio duration \\(2.00s\\)"):
            trim(samples, sample_rate)


class TestEndTrim:
    """Test cases for the EndTrim class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        trim = EndTrim(end_time=3.0)
        assert trim.end_time == 3.0
        assert trim.p == 1.0
        
        trim = EndTrim(end_time=2.5, p=0.8)
        assert trim.end_time == 2.5
        assert trim.p == 0.8
    
    def test_init_invalid_end_time_type(self):
        """Test initialization with invalid end_time type."""
        with pytest.raises(TypeError, match="end_time must be a number"):
            EndTrim(end_time="3.0")
    
    def test_init_non_positive_end_time(self):
        """Test initialization with non-positive end_time."""
        with pytest.raises(ValueError, match="end_time must be positive"):
            EndTrim(end_time=0.0)
        
        with pytest.raises(ValueError, match="end_time must be positive"):
            EndTrim(end_time=-1.0)
    
    def test_trim_basic_functionality(self):
        """Test basic end trimming functionality."""
        sample_rate = 44100
        samples = np.arange(5 * sample_rate, dtype=np.float32)  # 5 seconds
        
        trim = EndTrim(end_time=3.0)
        result = trim(samples, sample_rate)
        
        expected_end = int(3.0 * sample_rate)
        expected = samples[:expected_end]
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 3 * sample_rate  # 3 seconds
    
    def test_trim_end_time_exceeds_duration(self):
        """Test behavior when end_time exceeds audio duration."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)  # 2 seconds
        
        trim = EndTrim(end_time=5.0)  # 5 seconds > 2 seconds
        result = trim(samples, sample_rate)
        
        # Should return the full audio
        np.testing.assert_array_equal(result, samples)


class TestCenterTrim:
    """Test cases for the CenterTrim class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        trim = CenterTrim(duration=2.0)
        assert trim.duration == 2.0
        assert trim.p == 1.0
        
        trim = CenterTrim(duration=1.5, p=0.9)
        assert trim.duration == 1.5
        assert trim.p == 0.9
    
    def test_init_invalid_duration_type(self):
        """Test initialization with invalid duration type."""
        with pytest.raises(TypeError, match="duration must be a number"):
            CenterTrim(duration="2.0")
    
    def test_init_non_positive_duration(self):
        """Test initialization with non-positive duration."""
        with pytest.raises(ValueError, match="duration must be positive"):
            CenterTrim(duration=0.0)
        
        with pytest.raises(ValueError, match="duration must be positive"):
            CenterTrim(duration=-1.0)
    
    def test_trim_basic_functionality(self):
        """Test basic center trimming functionality."""
        sample_rate = 44100
        samples = np.arange(5 * sample_rate, dtype=np.float32)  # 5 seconds
        
        trim = CenterTrim(duration=2.0)
        result = trim(samples, sample_rate)
        
        # Center 2 seconds from 5 seconds: start at 1.5s, end at 3.5s
        expected_start = int(1.5 * sample_rate)
        expected_end = int(3.5 * sample_rate)
        expected = samples[expected_start:expected_end]
        
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 2 * sample_rate  # 2 seconds
    
    def test_trim_duration_exceeds_audio(self):
        """Test error when duration exceeds audio length."""
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)  # 2 seconds
        
        trim = CenterTrim(duration=3.0)  # 3 seconds > 2 seconds
        
        with pytest.raises(ValueError, match="duration \\(3.00s\\) exceeds audio duration \\(2.00s\\)"):
            trim(samples, sample_rate)


class TestIntegrationAndEdgeCases:
    """Integration tests and edge case testing."""
    
    def test_multichannel_audio(self):
        """Test trimming with multichannel audio."""
        sample_rate = 44100
        # Create stereo audio: 3 seconds, 2 channels
        samples = np.random.randn(3 * sample_rate, 2).astype(np.float32)
        
        trim = Trim(start_time=1.0, end_time=2.0)
        result = trim(samples, sample_rate)
        
        assert result.shape == (sample_rate, 2)  # 1 second, 2 channels
        
        # Verify the content is correct
        expected_start = int(1.0 * sample_rate)
        expected_end = int(2.0 * sample_rate)
        expected = samples[expected_start:expected_end]
        np.testing.assert_array_equal(result, expected)
    
    def test_different_sample_rates(self):
        """Test trimming with different sample rates."""
        for sample_rate in [22050, 44100, 48000]:
            samples = np.arange(2 * sample_rate, dtype=np.float32)
            
            trim = Trim(start_time=0.5, end_time=1.5)
            result = trim(samples, sample_rate)
            
            assert len(result) == sample_rate  # 1 second
    
    def test_very_short_audio(self):
        """Test trimming with very short audio."""
        sample_rate = 44100
        samples = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # 5 samples
        
        trim = Trim(start_time=0.0, end_time=2/sample_rate)  # ~2 samples
        result = trim(samples, sample_rate)
        
        assert len(result) == 2
        np.testing.assert_array_equal(result, samples[:2])
    
    def test_preserve_dtype(self):
        """Test that trimming preserves the original dtype."""
        sample_rate = 44100
        
        for dtype in [np.float32, np.float64, np.int16, np.int32]:
            samples = np.arange(sample_rate, dtype=dtype)  # 1 second
            
            trim = Trim(start_time=0.25, end_time=0.75)
            result = trim(samples, sample_rate)
            
            assert result.dtype == dtype
    
    @patch('random.random')
    def test_probability_behavior_across_classes(self, mock_random):
        """Test probability behavior is consistent across all trim classes."""
        mock_random.return_value = 0.8  # Greater than p=0.5
        
        sample_rate = 44100
        samples = np.arange(2 * sample_rate, dtype=np.float32)
        
        trim_classes = [
            Trim(start_time=0.5, p=0.5),
            RandomTrim(duration=1.0, p=0.5),
            StartTrim(start_time=0.5, p=0.5),
            EndTrim(end_time=1.5, p=0.5),
            CenterTrim(duration=1.0, p=0.5)
        ]
        
        for trim in trim_classes:
            result = trim(samples, sample_rate)
            # All should return original samples due to probability
            np.testing.assert_array_equal(result, samples)
    
    def test_zero_start_time(self):
        """Test behavior with zero start_time."""
        sample_rate = 44100
        samples = np.arange(sample_rate, dtype=np.float32)
        
        trim = Trim(start_time=0.0, end_time=0.5)
        result = trim(samples, sample_rate)
        
        expected = samples[:int(0.5 * sample_rate)]
        np.testing.assert_array_equal(result, expected)
    
    def test_boundary_sample_calculations(self):
        """Test that sample index calculations handle boundaries correctly."""
        sample_rate = 44100
        samples = np.arange(sample_rate, dtype=np.float32)  # 1 second
        
        # Test with non-integer sample boundaries
        trim = Trim(start_time=0.1, end_time=0.9)
        result = trim(samples, sample_rate)
        
        expected_start = int(0.1 * sample_rate)  # 4410
        expected_end = int(0.9 * sample_rate)    # 39690
        expected_length = expected_end - expected_start
        
        assert len(result) == expected_length
