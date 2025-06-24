import numpy as np
from soundmentations.core.transforms_interface import BaseTransform


class BaseLimiter(BaseTransform):
    """
    Base class for audio limiter transforms.
    
    This class provides the interface for applying a limiter to audio samples.
    It ensures that the transform can be applied with a specified probability.
    
    Parameters
    ----------
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.
    
    Notes
    -----
    This is an internal implementation detail. End users should use specific
    limiter classes like Limiter.
    """
    
    def __call__(self, samples: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply the limiter to the audio samples if the probability condition is met.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples as a 1D numpy array (mono audio only).
        sample_rate : int, optional
            Sample rate of the audio, by default 44100.
        
        Returns
        -------
        np.ndarray
            Audio samples after applying the limiter (or original if probability check fails).
        
        Raises
        ------
        TypeError
            If samples is not a numpy array or sample_rate is not an integer.
        ValueError
            If samples are empty, not 1D, or sample_rate is not positive.
        """
        self.validate_audio(samples, sample_rate)

        if not self.should_apply():
            return samples
        
        # Apply the specific limiter logic implemented by subclass
        return self._limit(samples, sample_rate)
    
    def _limit(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply the limiter effect to the audio samples.
        
        This method should be implemented by subclasses to define the specific
        limiting logic.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (validated by __call__).
        sample_rate : int
            Sample rate of the audio (validated by __call__).
        
        Returns
        -------
        np.ndarray
            Audio samples after applying the limiter.
        
        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Subclasses must implement this method with their specific limiting strategy.
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the _limit method")
    

class Limiter(BaseLimiter):
    """
    Apply hard limiting to audio samples to prevent clipping.
    
    This transform clips audio samples that exceed the specified threshold,
    preventing digital clipping and maintaining signal integrity within
    the specified dynamic range.
    
    Parameters
    ----------
    threshold : float, optional
        The threshold level for limiting, by default 0.9.
        Values above this threshold will be clipped. Must be between 0.0 and 1.0.
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.
    
    Examples
    --------
    Apply hard limiting to prevent clipping:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.amplitude import Limiter
    >>> 
    >>> # Create audio with some peaks above 0.9
    >>> audio = np.array([0.5, 1.2, -1.5, 0.8, 0.95])
    >>> 
    >>> # Apply limiting at 0.9 threshold
    >>> limiter = Limiter(threshold=0.9)
    >>> limited = limiter(audio, sample_rate=44100)
    >>> print(limited)  # [0.5, 0.9, -0.9, 0.8, 0.9]
    
    Use in audio processing pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Safe audio processing with limiting
    >>> safe_pipeline = S.Compose([
    ...     S.Gain(gain=12.0, p=1.0),           # Boost signal
    ...     S.Limiter(threshold=0.95, p=1.0),   # Prevent clipping
    ...     S.FadeOut(duration=0.1, p=0.5)      # Smooth ending
    ... ])
    >>> 
    >>> processed = safe_pipeline(audio, sample_rate=44100)
    
    Protect against digital distortion:
    
    >>> # Conservative limiting for pristine quality
    >>> conservative_limiter = Limiter(threshold=0.8, p=1.0)
    >>> clean_audio = conservative_limiter(loud_audio, sample_rate=44100)
    """
    
    def __init__(self, threshold: float = 0.9, p: float = 1.0):
        """
        Initialize the limiter transform.
        
        Parameters
        ----------
        threshold : float, optional
            Limiting threshold, by default 0.9. Must be between 0.01 and 1.0.
            Values below 0.01 may result in overly aggressive limiting.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        
        Raises
        ------
        TypeError
            If threshold is not a number.
        ValueError
            If threshold is not between 0.01 and 1.0.
        """
        super().__init__(p)
        
        if not isinstance(threshold, (float, int)):
            raise TypeError("threshold must be a number")
        if not (0.01 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0.01 and 1.0")
            
        self.threshold = threshold
    
    def _limit(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply hard limiting to the audio samples.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (validated by __call__).
        sample_rate : int
            Sample rate of the audio (not used in limiting but kept for consistency).
        
        Returns
        -------
        np.ndarray
            Audio samples with hard limiting applied.
        
        Notes
        -----
        Uses numpy.clip to apply hard limiting:
        - Positive values above threshold are clipped to +threshold
        - Negative values below -threshold are clipped to -threshold
        - Values within range remain unchanged
        """

        return np.clip(samples, -self.threshold, self.threshold)