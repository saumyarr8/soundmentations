import numpy as np
from soundmentations.core.transforms_interface import BaseTransform


class BaseFade(BaseTransform):
    """
    Base class for fade transforms.

    This class provides a framework for implementing fade-in and fade-out
    effects on audio samples. Subclasses should implement the specific
    fade logic.
    """

    def __call__(self, samples: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        self.validate_audio(samples, sample_rate)

        if not self.should_apply():
            return samples

        return self._fade(samples, sample_rate)

    def validate_duration_param(self, duration: float):
        """
        Validate duration parameter during initialization.
        
        Parameters
        ----------
        duration : float
            Duration value to validate.
            
        Raises
        ------
        TypeError
            If duration is not a number.
        ValueError
            If duration is not positive.
        """
        if not isinstance(duration, (float, int)):
            raise TypeError("duration must be a float or an integer")
        if duration <= 0:
            raise ValueError("duration must be positive")

    def _fade(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the _fade method")


class FadeIn(BaseFade):
    """
    Fade-in effect for audio samples.

    This transform applies a fade-in effect to the beginning of the audio samples.
    """
    def __init__(self, duration: float = 0.1, p: float = 1.0):
        """
        Initialize the FadeIn transform.

        Parameters
        ----------
        duration : float
            Duration of the fade-in effect in seconds (default is 0.1 seconds).
        p : float
            Probability of applying the fade-in effect (default is 1.0, always applies).
        """
        super().__init__(p)
        self.validate_duration_param(duration)
        self.duration = duration

    def _fade(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply fade-in effect to the audio samples.

        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (validated by __call__).
        sample_rate : int
            Sample rate of the audio (validated by __call__).  

        Returns
        -------
        np.ndarray
            Audio samples after applying the fade-in effect.
        """
        requested_fade_length = int(sample_rate * self.duration)
        fade_length = min(requested_fade_length, len(samples))

        # If fade covers entire audio, fade from zero to full amplitude
        fade_in_curve = np.linspace(0, 1, fade_length)
        faded_samples = samples.copy()
        faded_samples[:fade_length] *= fade_in_curve
        return faded_samples

class FadeOut(BaseFade):
    """
    Apply a fade-out effect to the end of audio samples.
    
    This transform gradually decreases the amplitude from full amplitude to
    silence (0) over the specified duration, creating a smooth fade-out effect.
    
    Parameters
    ----------
    duration : float, optional
        Duration of the fade-out effect in seconds, by default 0.1.
        Must be positive and less than the audio duration.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    """
    
    def __init__(self, duration: float = 0.1, p: float = 1.0):
        """
        Initialize the FadeOut transform.
        
        Parameters
        ----------
        duration : float, optional
            Duration of fade-out effect in seconds, by default 0.1.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        """
        super().__init__(p)
        self.validate_duration_param(duration)
        self.duration = duration

    def _fade(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply fade-out effect to the audio samples.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (validated by __call__).
        sample_rate : int
            Sample rate of the audio (validated by __call__).
            
        Returns
        -------
        np.ndarray
            Audio samples with fade-out effect applied.
        """
        requested_fade_length = int(sample_rate * self.duration)
        fade_length = min(requested_fade_length, len(samples))
        
        fade_out_curve = np.linspace(1, 0, fade_length)
        faded_samples = samples.copy()
        faded_samples[-fade_length:] *= fade_out_curve
        return faded_samples