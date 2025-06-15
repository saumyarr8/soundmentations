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
        
        Parameters:
        - pad_length (int): Target length for padding operations (in samples)
        - p (float): Probability of applying the transform. Default is 1.0.
        
        Raises:
        - TypeError: If pad_length is not an integer or p is not a float/int
        - ValueError: If pad_length is not positive or p is not between 0 and 1
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
    
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply the padding transform to the audio sample with probability p.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Padded audio samples (or original if probability check fails)
        
        Raises:
        - TypeError: If sample is not a numpy array
        - ValueError: If sample is empty or not 1D (mono audio only)
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
        
        Parameters:
        - sample (np.ndarray): Input audio samples (validated by __call__)
        
        Returns:
        - np.ndarray: Padded audio samples
        
        Note:
        - This method is called only after input validation and probability check
        - Subclasses must implement this method with their specific padding strategy
        """
        raise NotImplementedError("Subclasses must implement the _pad method")


class Pad(BasePad):
    """
    Pad audio to minimum length by adding zeros at the end.
    
    If the input audio is shorter than pad_length, zeros are appended to reach
    the minimum length. If already longer or equal, returns unchanged.
    
    Example:
        pad = Pad(pad_length=1000)
        # Input: [1, 2, 3] (length=3, pad_length=1000)
        # Output: [1, 2, 3, 0, 0, ..., 0] (length=1000)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at the end to reach minimum length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Audio with zeros added at end (if needed)
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
    
    Example:
        pad = CenterPad(pad_length=7)
        # Input: [1, 2, 3] (length=3, need 4 zeros)
        # Output: [0, 0, 1, 2, 3, 0, 0] (2 left, 2 right)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros symmetrically on both sides to reach minimum length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Audio with symmetric zero padding (if needed)
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
    
    Example:
        pad = StartPad(pad_length=1000)
        # Input: [1, 2, 3] (length=3, pad_length=1000)
        # Output: [0, 0, ..., 0, 1, 2, 3] (length=1000)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at the beginning to reach minimum length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Audio with zeros added at start (if needed)
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
    
    Example:
        pad = PadToLength(pad_length=5)
        # Input: [1, 2, 3] -> Output: [1, 2, 3, 0, 0]
        # Input: [1, 2, 3, 4, 5, 6, 7] -> Output: [1, 2, 3, 4, 5]
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Pad with zeros at end or trim from end to reach exact length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Audio with exactly pad_length samples
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
    
    Example:
        pad = CenterPadToLength(pad_length=7)
        # Input: [1, 2, 3] -> Output: [0, 0, 1, 2, 3, 0, 0]
        # Input: [1, 2, 3, 4, 5, 6, 7, 8, 9] -> Output: [2, 3, 4, 5, 6, 7, 8]
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Pad symmetrically or trim from center to reach exact length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Audio with exactly pad_length samples
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
    
    Example:
        pad = PadToMultiple(pad_length=1024)  # STFT frame size
        # Input: [audio...] (length=2050)
        # Output: [audio..., 0, 0, ..., 0] (length=3072 = 3*1024)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at end to make length a multiple of pad_length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples (mono audio only)
        
        Returns:
        - np.ndarray: Audio with length as multiple of pad_length
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


class CenterPad(BasePad):
    """
    Pad audio to minimum length by adding zeros symmetrically on both sides.
    
    If the input audio is shorter than pad_length, zeros are added equally
    to both sides. For odd padding amounts, the extra zero goes to the right.
    
    Example:
        pad = CenterPad(pad_length=7)
        # Input: [1, 2, 3] (length=3, need 4 zeros)
        # Output: [0, 0, 1, 2, 3, 0, 0] (2 left, 2 right)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros symmetrically on both sides to reach minimum length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples
        
        Returns:
        - np.ndarray: Audio with symmetric zero padding (if needed)
        """
        # Only pad if sample is shorter than target length
        if len(sample) < self.pad_length:
            # Calculate total padding needed
            total_padding = self.pad_length - len(sample)
            
            # Split padding between left and right sides
            left_pad = total_padding // 2  # Integer division
            right_pad = total_padding - left_pad  # Handles odd numbers
            
            # Create zero arrays with same shape as input (preserve channels)
            if sample.ndim == 1:
                left_zeros = np.zeros(left_pad, dtype=sample.dtype)
                right_zeros = np.zeros(right_pad, dtype=sample.dtype)
            else:
                # For multichannel audio, preserve the channel dimension
                left_shape = (left_pad,) + sample.shape[1:]
                right_shape = (right_pad,) + sample.shape[1:]
                left_zeros = np.zeros(left_shape, dtype=sample.dtype)
                right_zeros = np.zeros(right_shape, dtype=sample.dtype)
            
            # Concatenate: [left_zeros][sample][right_zeros]
            return np.concatenate((left_zeros, sample, right_zeros), axis=0)
        
        # Return unchanged if already long enough
        return sample


class StartPad(BasePad):
    """
    Pad audio to minimum length by adding zeros at the beginning.
    
    If the input audio is shorter than pad_length, zeros are prepended to reach
    the minimum length. If already longer or equal, returns unchanged.
    
    Example:
        pad = StartPad(pad_length=1000)
        # Input: [1, 2, 3] (length=3, pad_length=1000)
        # Output: [0, 0, ..., 0, 1, 2, 3] (length=1000)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at the beginning to reach minimum length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples
        
        Returns:
        - np.ndarray: Audio with zeros added at start (if needed)
        """
        # Only pad if sample is shorter than target length
        if len(sample) < self.pad_length:
            # Calculate how many zeros to add
            padding_needed = self.pad_length - len(sample)
            # Create zero padding with same shape as input (preserve channels)
            if sample.ndim == 1:
                padding = np.zeros(padding_needed, dtype=sample.dtype)
            else:
                # For multichannel audio, preserve the channel dimension
                padding_shape = (padding_needed,) + sample.shape[1:]
                padding = np.zeros(padding_shape, dtype=sample.dtype)
            # Prepend padding to start: [zeros][sample]
            return np.concatenate((padding, sample), axis=0)
        
        # Return unchanged if already long enough
        return sample


class PadToLength(BasePad):
    """
    Pad or trim audio to exact target length using end operations.
    
    - If shorter: adds zeros at the end to reach exact length
    - If longer: trims from the end to reach exact length
    - If equal: returns unchanged
    
    Example:
        pad = PadToLength(pad_length=5)
        # Input: [1, 2, 3] -> Output: [1, 2, 3, 0, 0]
        # Input: [1, 2, 3, 4, 5, 6, 7] -> Output: [1, 2, 3, 4, 5]
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Pad with zeros at end or trim from end to reach exact length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples
        
        Returns:
        - np.ndarray: Audio with exactly pad_length samples
        """
        current_length = len(sample)
        
        if current_length < self.pad_length:
            # Sample is too short - add zeros at end
            padding_needed = self.pad_length - current_length
            # Create zero padding with same shape as input (preserve channels)
            if sample.ndim == 1:
                padding = np.zeros(padding_needed, dtype=sample.dtype)
            else:
                # For multichannel audio, preserve the channel dimension
                padding_shape = (padding_needed,) + sample.shape[1:]
                padding = np.zeros(padding_shape, dtype=sample.dtype)
            return np.concatenate((sample, padding), axis=0)
        
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
    
    Example:
        pad = CenterPadToLength(pad_length=7)
        # Input: [1, 2, 3] -> Output: [0, 0, 1, 2, 3, 0, 0]
        # Input: [1, 2, 3, 4, 5, 6, 7, 8, 9] -> Output: [2, 3, 4, 5, 6, 7, 8]
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Pad symmetrically or trim from center to reach exact length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples
        
        Returns:
        - np.ndarray: Audio with exactly pad_length samples
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
    
    Example:
        pad = PadToMultiple(pad_length=1024)  # STFT frame size
        # Input: [audio...] (length=2050)
        # Output: [audio..., 0, 0, ..., 0] (length=3072 = 3*1024)
    """
    
    def _pad(self, sample: np.ndarray) -> np.ndarray:
        """
        Add zeros at end to make length a multiple of pad_length.
        
        Parameters:
        - sample (np.ndarray): Input audio samples
        
        Returns:
        - np.ndarray: Audio with length as multiple of pad_length
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