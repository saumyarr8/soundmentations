import random
import numpy as np


class BaseTransform:
    """
    Internal base class for all audio transforms in Soundmentations.
    
    .. warning::
        This class is for internal use only and should not be used directly
        by end users. Use the specific transform classes instead.
    
    This class provides common functionality for all transforms including
    probability handling and validation. All transform classes inherit
    from this base class to ensure consistent behavior across the library.
    
    Parameters
    ----------
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.
    """
    
    def __init__(self, p: float = 1.0):
        if not isinstance(p, (float, int)):
            raise TypeError("p must be a float or an integer")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0.0 and 1.0")
        self.p = p

    def should_apply(self) -> bool:
        """
        Determine whether the transform should be applied based on probability.
        
        Returns
        -------
        bool
            True if the transform should be applied, False otherwise.
        """
        return random.random() < self.p

    def validate_audio(self, samples: np.ndarray, sample_rate: int):
        """
        Validate audio samples and sample rate.

        Parameters
        ----------
        samples : np.ndarray
            Audio samples as a 1D numpy array.
        sample_rate : int
            Sample rate of the audio.

        Raises
        ------
        TypeError
            If types are incorrect.
        ValueError
            If array is empty, not 1D, or sample rate is non-positive.
        """
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy array")
        if samples.size == 0:
            raise ValueError("Input samples cannot be empty")
        if samples.ndim != 1:
            raise ValueError("samples must be a 1D array (mono audio only)")
        if not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__.")