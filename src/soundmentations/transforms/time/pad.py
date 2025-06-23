import numpy as np
import random

class BasePad:
    """
    Base class for audio padding transforms.
    
    This class provides common functionality for padding operations including
    parameter validation and probability handling. All padding subclasses
    inherit from this base class and implement the _pad method.
    """
    
    def __init__(self, pad_length: int, p: float = 1.0):
        """
        Initialize the base padding transform.
        
        Parameters
        ----------
        pad_length : int
            Target length for padding operations (in samples).
            Must be a positive integer.
        p : float, optional
            Probability of applying the transform, by default 1.0.
            Must be between 0.0 and 1.0.
        
        Raises
        ------
        TypeError
            If pad_length is not an integer or p is not a float/int.
        ValueError
            If pad_length is not positive or p is not between 0 and 1.
        """
        # Validate pad_length type and value
        if not isinstance(pad_length, int):
            raise TypeError("pad_length must be an integer")
        if pad_length <= 0:
            raise ValueError("pad_length must be positive")
            
        # Validate probability p
        if not isinstance(p, (float, int)):
            raise TypeError("p must be a float or an integer")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0.0 and 1.0")
            
        self.p = p
        self.pad_length = pad_length

    def __call__(self, sample: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply the padding transform to the audio sample with probability p.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples as a 1D numpy array (mono audio only).
        
        Returns
        -------
        np.ndarray
            Padded audio samples (or original if probability check fails).
        
        Raises
        ------
        TypeError
            If sample is not a numpy array.
        ValueError
            If sample is empty or not 1D (mono audio only).
        """
        # Validate input sample
        if not isinstance(sample, np.ndarray):
            raise TypeError("sample must be a numpy array")
        if sample.size == 0:
            raise ValueError("sample cannot be empty")
        if sample.ndim != 1:
            raise ValueError("sample must be a 1D array (mono audio only)")
        
        # Apply probability check - skip transformation if random value exceeds p
        if random.random() > self.p:
            return sample
            
        # Apply the specific padding logic implemented by subclass
        return self._pad(sample)
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Abstract method to be implemented by subclasses for specific padding logic.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (validated by __call__).
        
        Returns
        -------
        np.ndarray
            Padded audio samples.
        
        Notes
        -----
        This method is called only after input validation and probability check.
        Subclasses must implement this method with their specific padding strategy.
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the _pad method")


class Pad(BasePad):
    """
    Pad audio to minimum length by adding zeros at the end.
    
    If the input audio is shorter than pad_length, zeros are appended to reach
    the minimum length. If already longer or equal, returns unchanged.
    
    Parameters
    ----------
    pad_length : int
        Minimum length for the audio in samples.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Apply end padding to ensure minimum length:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import Pad
    >>> 
    >>> # Create short audio sample
    >>> audio = np.array([0.1, 0.2, 0.3])
    >>> 
    >>> # Pad to minimum 1000 samples
    >>> pad_transform = Pad(pad_length=1000)
    >>> padded = pad_transform(audio)
    >>> print(len(padded))  # 1000
    
    Use in a pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Ensure all audio is at least 2 seconds (44.1kHz)
    >>> augment = S.Compose([
    ...     S.Pad(pad_length=88200, p=1.0),
    ...     S.Gain(gain=3.0, p=0.5)
    ... ])
    >>> 
    >>> result = augment(audio)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at the end to reach minimum length.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (mono audio only).
        
        Returns
        -------
        np.ndarray
            Audio with zeros added at end (if needed).
        """
        # Only pad if sample is shorter than target length
        if len(sample) < self.pad_length:
            # Calculate how many zeros to add
            padding_needed = self.pad_length - len(sample)
            # Create zero padding with same dtype as input
            padding = np.zeros(padding_needed, dtype=sample.dtype)
            # Append padding to end: [sample][zeros]
            return np.concatenate((sample, padding))
        
        # Return unchanged if already long enough
        return sample


class CenterPad(BasePad):
    """
    Pad audio to minimum length by adding zeros symmetrically on both sides.
    
    If the input audio is shorter than pad_length, zeros are added equally
    to both sides. For odd padding amounts, the extra zero goes to the right.
    
    Parameters
    ----------
    pad_length : int
        Minimum length for the audio in samples.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Apply symmetric padding:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import CenterPad
    >>> 
    >>> audio = np.array([1, 2, 3])
    >>> pad_transform = CenterPad(pad_length=7)
    >>> result = pad_transform(audio)
    >>> print(result)  # [0 0 1 2 3 0 0]
    
    Use for centering audio in fixed-length windows:
    
    >>> # Center audio in 5-second windows (44.1kHz)
    >>> center_pad = CenterPad(pad_length=220500)
    >>> centered_audio = center_pad(audio_sample)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros symmetrically on both sides to reach minimum length.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (mono audio only).
        
        Returns
        -------
        np.ndarray
            Audio with symmetric zero padding (if needed).
        """
        # Only pad if sample is shorter than target length
        if len(sample) < self.pad_length:
            # Calculate total padding needed
            total_padding = self.pad_length - len(sample)
            
            # Split padding between left and right sides
            left_pad = total_padding // 2  # Integer division
            right_pad = total_padding - left_pad  # Handles odd numbers
            
            # Create zero arrays with same dtype as input
            left_zeros = np.zeros(left_pad, dtype=sample.dtype)
            right_zeros = np.zeros(right_pad, dtype=sample.dtype)
            
            # Concatenate: [left_zeros][sample][right_zeros]
            return np.concatenate((left_zeros, sample, right_zeros))
        
        # Return unchanged if already long enough
        return sample


class StartPad(BasePad):
    """
    Pad audio to minimum length by adding zeros at the beginning.
    
    If the input audio is shorter than pad_length, zeros are prepended to reach
    the minimum length. If already longer or equal, returns unchanged.
    
    Parameters
    ----------
    pad_length : int
        Minimum length for the audio in samples.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Apply start padding:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import StartPad
    >>> 
    >>> audio = np.array([1, 2, 3])
    >>> pad_transform = StartPad(pad_length=6)
    >>> result = pad_transform(audio)
    >>> print(result)  # [0 0 0 1 2 3]
    
    Use for aligning audio to end of fixed windows:
    
    >>> # Align audio to end of 3-second windows
    >>> start_pad = StartPad(pad_length=132300)  # 3 seconds at 44.1kHz
    >>> aligned_audio = start_pad(audio_sample)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at the beginning to reach minimum length.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (mono audio only).
        
        Returns
        -------
        np.ndarray
            Audio with zeros added at start (if needed).
        """
        # Only pad if sample is shorter than target length
        if len(sample) < self.pad_length:
            # Calculate how many zeros to add
            padding_needed = self.pad_length - len(sample)
            # Create zero padding with same dtype as input
            padding = np.zeros(padding_needed, dtype=sample.dtype)
            # Prepend padding to start: [zeros][sample]
            return np.concatenate((padding, sample))
        
        # Return unchanged if already long enough
        return sample


class PadToLength(BasePad):
    """
    Pad or trim audio to exact target length using end operations.
    
    - If shorter: adds zeros at the end to reach exact length
    - If longer: trims from the end to reach exact length
    - If equal: returns unchanged
    
    Parameters
    ----------
    pad_length : int
        Exact target length for the audio in samples.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Normalize all audio to exact length:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import PadToLength
    >>> 
    >>> # Short audio
    >>> short_audio = np.array([1, 2, 3])
    >>> # Long audio
    >>> long_audio = np.arange(10)
    >>> 
    >>> pad_transform = PadToLength(pad_length=5)
    >>> 
    >>> result1 = pad_transform(short_audio)
    >>> print(result1)  # [1 2 3 0 0]
    >>> 
    >>> result2 = pad_transform(long_audio)
    >>> print(result2)  # [0 1 2 3 4]
    
    Use for fixed-length model inputs:
    
    >>> # Ensure all audio is exactly 2 seconds for ML model
    >>> normalize_length = PadToLength(pad_length=88200)  # 2s at 44.1kHz
    >>> model_input = normalize_length(variable_length_audio)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Pad with zeros at end or trim from end to reach exact length.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (mono audio only).
        
        Returns
        -------
        np.ndarray
            Audio with exactly pad_length samples.
        """
        current_length = len(sample)
        
        if current_length < self.pad_length:
            # Sample is too short - add zeros at end
            padding_needed = self.pad_length - current_length
            padding = np.zeros(padding_needed, dtype=sample.dtype)
            return np.concatenate((sample, padding))
        
        elif current_length > self.pad_length:
            # Sample is too long - trim from end, keep first pad_length samples
            return sample[:self.pad_length]
        
        else:
            # Sample is already the correct length
            return sample


class CenterPadToLength(BasePad):
    """
    Pad or trim audio to exact target length using center operations.
    
    - If shorter: adds zeros symmetrically on both sides
    - If longer: trims symmetrically from both sides (keeps center)
    - If equal: returns unchanged
    
    Parameters
    ----------
    pad_length : int
        Exact target length for the audio in samples.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Center-normalize audio to exact length:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import CenterPadToLength
    >>> 
    >>> # Short audio - will be center-padded
    >>> short_audio = np.array([1, 2, 3])
    >>> # Long audio - will be center-trimmed
    >>> long_audio = np.arange(9)
    >>> 
    >>> pad_transform = CenterPadToLength(pad_length=7)
    >>> 
    >>> result1 = pad_transform(short_audio)
    >>> print(result1)  # [0 0 1 2 3 0 0]
    >>> 
    >>> result2 = pad_transform(long_audio)
    >>> print(result2)  # [1 2 3 4 5 6 7]
    
    Use for preserving important audio content in center:
    
    >>> # Keep center 3 seconds for speech processing
    >>> center_normalize = CenterPadToLength(pad_length=132300)
    >>> processed_audio = center_normalize(speech_audio)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Pad symmetrically or trim from center to reach exact length.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (mono audio only).
        
        Returns
        -------
        np.ndarray
            Audio with exactly pad_length samples.
        """
        current_length = len(sample)
        
        if current_length < self.pad_length:
            # Sample is too short - center pad (same logic as CenterPad)
            total_padding = self.pad_length - current_length
            left_pad = total_padding // 2
            right_pad = total_padding - left_pad
            
            left_zeros = np.zeros(left_pad, dtype=sample.dtype)
            right_zeros = np.zeros(right_pad, dtype=sample.dtype)
            return np.concatenate((left_zeros, sample, right_zeros))
        
        elif current_length > self.pad_length:
            # Sample is too long - center trim (keep middle portion)
            excess = current_length - self.pad_length
            start_trim = excess // 2  # How much to remove from start
            end_index = start_trim + self.pad_length  # Where to stop
            return sample[start_trim:end_index]
        
        else:
            # Sample is already the correct length
            return sample


class PadToMultiple(BasePad):
    """
    Pad audio to make its length a multiple of the specified value.
    
    This is useful for STFT operations where frame sizes must be multiples
    of certain values. Only adds padding at the end, never trims.
    
    Parameters
    ----------
    pad_length : int
        The multiple value. Audio length will be padded to next multiple of this value.
        Common values: 1024, 512, 256 for STFT operations.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Pad for STFT-friendly lengths:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import PadToMultiple
    >>> 
    >>> # Audio with length 2050 samples
    >>> audio = np.random.randn(2050)
    >>> 
    >>> # Pad to multiple of 1024 (STFT frame size)
    >>> pad_transform = PadToMultiple(pad_length=1024)
    >>> result = pad_transform(audio)
    >>> print(len(result))  # 3072 (3 * 1024)
    
    Use in spectral processing pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Prepare audio for spectral analysis
    >>> spectral_prep = S.Compose([
    ...     S.PadToMultiple(pad_length=512, p=1.0),  # STFT-friendly
    ...     S.Gain(gain=(-3, 3), p=0.5)
    ... ])
    >>> 
    >>> stft_ready_audio = spectral_prep(raw_audio)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at end to make length a multiple of pad_length.
        
        Parameters
        ----------
        sample : np.ndarray
            Input audio samples (mono audio only).
        
        Returns
        -------
        np.ndarray
            Audio with length as multiple of pad_length.
        
        Notes
        -----
        If the input length is already a multiple of pad_length,
        no padding is applied and the original array is returned.
        """
        current_length = len(sample)
        
        # Calculate remainder when dividing by target multiple
        remainder = current_length % self.pad_length
        
        if remainder == 0:
            # Already a perfect multiple, no padding needed
            return sample
        
        # Calculate how much padding needed to reach next multiple
        padding_needed = self.pad_length - remainder
        
        # Add zeros at the end
        padding = np.zeros(padding_needed, dtype=sample.dtype)
        return np.concatenate((sample, padding))