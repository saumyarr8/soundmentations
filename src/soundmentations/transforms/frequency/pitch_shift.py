import numpy as np
import librosa
from soundmentations.core.transforms_interface import BaseTransform


class BasePitchShift(BaseTransform):
    """
    Base class for pitch shifting transforms.

    This class provides input validation and probability application.
    Subclasses must implement the _shift method.
    """

    def __call__(self, samples: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        self.validate_audio(samples, sample_rate)

        if not self.should_apply():
            return samples

        return self._shift(samples, sample_rate)
    
    def validate_semitones(self, semitones: float):
        """
        Validate the semitones parameter for pitch shifting.
        
        Parameters
        ----------
        semitones : float
            Number of semitones to shift.
        
        Raises
        ------
        TypeError
            If semitones is not a number.
        ValueError
            If semitones is outside the range of -48 to 48 (±4 octaves).
        """
        if not isinstance(semitones, (float, int)):
            raise TypeError("semitones must be a number")
        if not (-48 <= semitones <= 48):
            raise ValueError("semitones must be between -48 and 48 (±4 octaves)")

    def _shift(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _shift method")


class PitchShift(BasePitchShift):
    """
    Shift the pitch of audio by a specified number of semitones.
    
    Parameters
    ----------
    semitones : float
        Number of semitones to shift (positive or negative).
        - 12 semitones = 1 octave
        - Positive: pitch up, Negative: pitch down
    p : float, optional
        Probability of applying the transform, by default 1.0.
    """
    
    def __init__(self, semitones: float, p: float = 1.0):
        super().__init__(p)
        
        # Use the base class validation method
        self.validate_semitones(semitones)
        
        self.semitones = semitones
    
    def _shift(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply pitch shifting using librosa with error handling."""
        # Skip processing if no shift needed
        if abs(self.semitones) < 0.01:
            return samples
        
        try:
            return librosa.effects.pitch_shift(
                y=samples, 
                sr=sample_rate, 
                n_steps=self.semitones
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Pitch shifting failed: {e}. Returning original audio.")
            return samples


class RandomPitchShift(BasePitchShift):
    """
    Randomly shift the pitch within a specified semitone range.
    
    This class wraps PitchShift to provide random pitch variations
    for data augmentation purposes.

    Parameters
    ----------
    min_semitones : float, optional
        Minimum semitone shift, by default -2.0.
    max_semitones : float, optional  
        Maximum semitone shift, by default 2.0.
    p : float, optional
        Probability of applying the transform, by default 1.0.
        
    Examples
    --------
    >>> # Random pitch variation for training data
    >>> random_pitch = RandomPitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.8)
    >>> augmented = random_pitch(audio, sample_rate=44100)
    """

    def __init__(self, min_semitones: float = -2.0, max_semitones: float = 2.0, p: float = 1.0):
        super().__init__(p)
        
        self.validate_semitones(min_semitones)
        self.validate_semitones(max_semitones)
        
        if min_semitones > max_semitones:
            raise ValueError("min_semitones must be <= max_semitones")
        
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def _shift(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Implement the _shift method required by BasePitchShift.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples.
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Audio samples with random pitch shift applied.
        """
        random_semitones = np.random.uniform(self.min_semitones, self.max_semitones)
        
        # Skip processing if no shift needed
        if abs(random_semitones) < 0.01:
            return samples
        
        try:
            return librosa.effects.pitch_shift(
                y=samples, 
                sr=sample_rate, 
                n_steps=random_semitones
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Pitch shifting failed: {e}. Returning original audio.")
            return samples