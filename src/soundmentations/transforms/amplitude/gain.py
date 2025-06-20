import numpy as np
import random

class BaseGain:
    """
    Base class for audio gain transforms.
    
    This class provides common functionality for gain operations including
    parameter validation and probability handling. All gain subclasses
    inherit from this base class and implement the _gain method.
    """
    
    def __init__(self, p: float = 1.0):
        """
        Initialize the base gain transform.
        
        Parameters
        ----------
        p : float, optional
            Probability of applying the gain transform, by default 1.0 (always apply).
            Must be between 0.0 and 1.0.
            
        Raises
        ------
        TypeError
            If p is not a float or int.
        ValueError
            If p is not between 0 and 1.
        """
        if not isinstance(p, (float, int)):
            raise TypeError("p must be a float or int.")
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1.")
        self.p = p

    def __call__(self, samples: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply the gain transform to the audio samples with probability p.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples as a 1D numpy array.
        sample_rate : int, optional
            Sample rate of the audio, by default 44100.
            
        Returns
        -------
        np.ndarray
            Transformed audio samples (or original if probability check fails).
            
        Raises
        ------
        ValueError
            If input samples are empty.
        """
        if random.random() > self.p:
            return samples
        if samples.size == 0:
            raise ValueError("Input samples cannot be empty.")
        return self._gain(samples)
        
    def _gain(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply gain to the audio samples.
        
        This is an abstract method that must be implemented by subclasses
        to define the specific gain operation.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (validated by __call__).
            
        Returns
        -------
        np.ndarray
            Audio samples after applying gain.
            
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class Gain(BaseGain):
    """
    Apply a fixed gain (in dB) to audio samples.
    
    This transform multiplies the audio samples by a gain factor derived
    from the specified gain in decibels. Optionally clips the output to
    prevent values from exceeding the [-1, 1] range.
    
    Parameters
    ----------
    gain : float, optional
        Gain in decibels, by default 1.0. Positive values increase volume,
        negative values decrease volume.
    clip : bool, optional
        Whether to clip the output to [-1, 1] range, by default True.
        Prevents audio distortion from excessive gain.
    p : float, optional
        Probability of applying the gain transform, by default 1.0.
        
    Examples
    --------
    Apply a fixed gain to audio samples:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.amplitude import Gain
    >>> 
    >>> # Create audio samples
    >>> samples = np.array([0.1, 0.2, -0.1, 0.3])
    >>> 
    >>> # Apply +6dB gain
    >>> gain_transform = Gain(gain=6.0)
    >>> amplified = gain_transform(samples)
    >>> 
    >>> # Apply -12dB gain with 50% probability
    >>> quiet_transform = Gain(gain=-12.0, p=0.5)
    >>> result = quiet_transform(samples)
    
    Use in a pipeline with other transforms:
    
    >>> import soundmentations as S
    >>> 
    >>> # Create augmentation pipeline
    >>> augment = S.Compose([
    ...     S.RandomTrim(duration=(1.0, 3.0), p=0.8),
    ...     S.Gain(gain=6.0, clip=True, p=0.7),
    ...     S.PadToLength(pad_length=44100, p=0.5)
    ... ])
    >>> 
    >>> # Apply pipeline to audio
    >>> audio_samples = np.random.randn(22050)  # 0.5 seconds at 44.1kHz
    >>> augmented = augment(samples=audio_samples, sample_rate=44100)
    
    Different gain scenarios:
    
    >>> # Boost quiet audio
    >>> boost = Gain(gain=12.0, clip=True)
    >>> 
    >>> # Attenuate loud audio
    >>> attenuate = Gain(gain=-6.0, clip=False)
    >>> 
    >>> # Random volume variation
    >>> random_volume = Gain(gain=np.random.uniform(-10, 10), p=0.6)
    """
    
    def __init__(self, gain: float = 1.0, clip: bool = True, p: float = 1.0):
        """
        Initialize the Gain transform.
        
        Parameters
        ----------
        gain : float, optional
            Gain in decibels, by default 1.0.
        clip : bool, optional
            Whether to clip the output to [-1, 1] range, by default True.
        p : float, optional
            Probability of applying the gain transform, by default 1.0.
        """
        super().__init__(p)
        self.gain = gain
        self.gain_factor = 10 ** (gain / 20)
        self.clip = clip

    def _gain(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply gain to the audio samples.
        
        Multiplies the input samples by the gain factor and optionally
        clips the result to the [-1, 1] range.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples.
            
        Returns
        -------
        np.ndarray
            Audio samples after applying gain and optional clipping.
        """
        result = samples * self.gain_factor
        if self.clip:
            result = np.clip(result, -1.0, 1.0)
        return result