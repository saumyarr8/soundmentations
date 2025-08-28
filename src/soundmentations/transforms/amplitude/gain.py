"""
Audio gain transforms for the soundmentations library.

This module provides various gain-based audio transformations for audio
augmentation and processing. All transforms apply gain modifications in
decibel (dB) scale and convert to linear scale for audio processing.

Classes
-------
BaseGain
    Abstract base class for all gain transforms
Gain
    Apply a fixed gain to audio samples
RandomGain
    Apply a random gain within a specified range
PerSampleRandomGain
    Apply different random gains to each sample in a batch
RandomGainEnvelope
    Apply a smoothly varying random gain envelope

Examples
--------
Basic usage:

>>> import numpy as np
>>> from soundmentations.transforms.amplitude.gain import Gain, RandomGain
>>> 
>>> # Create test audio
>>> audio = np.random.randn(1000) * 0.1
>>> 
>>> # Apply fixed gain
>>> gain_transform = Gain(gain=6.0)
>>> boosted = gain_transform(audio)
>>> 
>>> # Apply random gain
>>> random_gain = RandomGain(min_gain=-6.0, max_gain=6.0)
>>> varied = random_gain(audio)

Notes
-----
- All gain values are specified in decibels (dB)
- Positive gain values increase volume, negative values decrease volume
- 6 dB gain approximately doubles the amplitude
- -6 dB gain approximately halves the amplitude
- 20 dB gain increases amplitude by 10x
- Clipping is available to prevent values exceeding [-1, 1] range
"""

import numpy as np
import random


class BaseGain:
    """
    Base class for audio gain transforms.
    
    This abstract base class provides common functionality for all gain-based
    audio transformations including parameter validation, probability handling,
    and the core application logic. All gain subclasses inherit from this base
    class and must implement the abstract `_gain` method.
    
    The base class handles:
    - Probability-based transform application
    - Input validation (empty arrays, etc.)
    - Common parameter validation (probability values)
    - Clipping behavior configuration
    
    Parameters
    ----------
    clip : bool, optional
        Whether to clip the output to [-1, 1] range, by default True.
        This prevents audio distortion from excessive gain values.
    p : float, optional
        Probability of applying the gain transform, by default 1.0 (always apply).
        Must be between 0.0 and 1.0. When p < 1.0, the transform is applied
        randomly with the specified probability.
    
    Raises
    ------
    TypeError
        If p is not a float or int.
    ValueError
        If p is not between 0 and 1.
    NotImplementedError
        If `_gain` method is called on the base class directly.
    
    Notes
    -----
    This is an abstract base class and should not be instantiated directly.
    Use concrete subclasses like Gain, RandomGain, etc.
    
    The probability mechanism uses Python's random.random() for consistency
    across the audio processing pipeline.
    
    See Also
    --------
    Gain : Apply a fixed gain to audio samples
    RandomGain : Apply a random gain within a specified range
    PerSampleRandomGain : Apply different random gains to each sample
    RandomGainEnvelope : Apply a smoothly varying gain envelope
    """
    
    def __init__(self, clip: bool = True, p: float = 1.0):
        """
        Initialize the base gain transform.
        
        Parameters
        ----------
        clip : bool, optional
            Whether to clip the output to [-1, 1] range, by default True.
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
        self.clip = clip  # Use the parameter instead of hardcoding True

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
        super().__init__(clip=clip, p=p)
        self.gain = gain
        self.gain_factor = 10 ** (gain / 20)

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
    

class RandomGain(Gain):
    """
    Apply a random gain to audio samples within a specified range.
    
    This transform randomly selects a gain value from a uniform distribution
    between min_gain and max_gain, applying it to the audio samples.
    
    Parameters
    ----------
    min_gain : float
        Minimum gain in decibels.
    max_gain : float
        Maximum gain in decibels.
    clip : bool, optional
        Whether to clip the output to [-1, 1] range, by default True.
    p : float, optional
        Probability of applying the random gain transform, by default 1.0.
    
    Examples
    --------
    >>> import numpy as np
    >>> from soundmentations.transforms.amplitude import RandomGain
    >>> 
    >>> # Create audio samples
    >>> samples = np.array([0.1, 0.2, -0.1, 0.3])
    >>> 
    >>> # Apply random gain between -6dB and +6dB
    >>> random_gain_transform = RandomGain(min_gain=-6.0, max_gain=6.0)
    >>> result = random_gain_transform(samples)
    """
    
    def __init__(self, min_gain: float, max_gain: float, clip: bool = True, p: float = 1.0):
        super().__init__(gain=0.0, clip=clip, p=p)
        if min_gain > max_gain:
            raise ValueError("min_gain must be less than or equal to max_gain")
        self.min_gain = min_gain
        self.max_gain = max_gain

    def _gain(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply a random gain to the audio samples.
        
        Selects a random gain value from the specified range and applies it.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples.
            
        Returns
        -------
        np.ndarray
            Audio samples after applying the random gain and optional clipping.
        """
        random_gain_value = np.random.uniform(self.min_gain, self.max_gain)
        self.gain_factor = 10 ** (random_gain_value / 20)
        return super()._gain(samples)
    

class PerSampleRandomGain(Gain):
    """
    Apply a different random gain to each audio sample in a batch.
    
    This transform applies a unique random gain value, drawn from a uniform
    distribution between min_gain and max_gain, to each sample in the input batch.
    This is useful for batch processing where you want different gain variations
    for each audio sample in the batch, creating diverse augmentations.
    
    Parameters
    ----------
    min_gain : float
        Minimum gain in decibels for the random range.
    max_gain : float
        Maximum gain in decibels for the random range.
    clip : bool, optional
        Whether to clip the output to [-1, 1] range, by default True.
    p : float, optional
        Probability of applying the per-sample random gain transform, by default 1.0.
    
    Examples
    --------
    Basic batch processing:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.amplitude import PerSampleRandomGain
    >>> 
    >>> # Create batch of audio samples (2 samples, each 1000 samples long)
    >>> batch_samples = np.random.randn(2, 1000) * 0.1
    >>> 
    >>> # Apply different random gain to each sample in batch
    >>> per_sample_transform = PerSampleRandomGain(min_gain=-6.0, max_gain=6.0)
    >>> result = per_sample_transform(batch_samples)
    >>> 
    >>> # Each row now has a different gain applied
    >>> print(f"Sample 1 max: {np.max(np.abs(result[0])):.3f}")
    >>> print(f"Sample 2 max: {np.max(np.abs(result[1])):.3f}")
    
    Machine learning data augmentation:
    
    >>> # Training data preparation with varied gains
    >>> ml_augment = PerSampleRandomGain(
    ...     min_gain=-12.0, 
    ...     max_gain=12.0, 
    ...     clip=True, 
    ...     p=0.8
    ... )
    >>> 
    >>> # Process batch for training
    >>> training_batch = np.random.randn(32, 16000)  # 32 samples, 16k each
    >>> augmented_batch = ml_augment(training_batch)
    
    Different use cases:
    
    >>> # Subtle variations for speech data
    >>> speech_augment = PerSampleRandomGain(min_gain=-3.0, max_gain=3.0)
    >>> 
    >>> # Dramatic variations for sound effects
    >>> sfx_augment = PerSampleRandomGain(min_gain=-20.0, max_gain=10.0)
    >>> 
    >>> # Conservative augmentation with low probability
    >>> conservative_augment = PerSampleRandomGain(
    ...     min_gain=-1.5, max_gain=1.5, p=0.3
    ... )
    
    Notes
    -----
    - Requires 2D input arrays where first dimension is batch size
    - Each sample in the batch gets an independent random gain
    - Useful for creating diverse training data in machine learning
    - The transform maintains the batch structure and sample lengths
    - Random gains are independently sampled for each batch item
    
    Raises
    ------
    ValueError
        If input is not a 2D array or if min_gain > max_gain
    
    See Also
    --------
    RandomGain : Apply a single random gain to entire audio
    Gain : Apply a fixed gain to audio samples
    RandomGainEnvelope : Apply a smoothly varying gain envelope
    """
    
    def __init__(self, min_gain: float, max_gain: float, clip: bool = True, p: float = 1.0):
        super().__init__(gain=0.0, clip=clip, p=p)
        if min_gain > max_gain:
            raise ValueError("min_gain must be less than or equal to max_gain")
        self.min_gain = min_gain
        self.max_gain = max_gain

    def _gain(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply a different random gain to each audio sample in the batch.
        
        Parameters
        ----------
        samples : np.ndarray
            Input batch of audio samples (2D array).
            
        Returns
        -------
        np.ndarray
            Audio samples after applying per-sample random gains and optional clipping.
            
        Raises
        ------
        ValueError
            If input samples are not a 2D array.
        """
        if samples.ndim != 2:
            raise ValueError("Input samples must be a 2D array for per-sample processing.")
        
        result = np.empty_like(samples)
        for i in range(samples.shape[0]):
            random_gain_value = np.random.uniform(self.min_gain, self.max_gain)
            self.gain_factor = 10 ** (random_gain_value / 20)
            result[i] = super()._gain(samples[i])
        return result
    
class RandomGainEnvelope(Gain):
    """
    Apply a smoothly varying random gain envelope to audio samples.
    
    This transform creates a smooth gain envelope by generating random gain values
    at control points and interpolating between them. This results in gradual
    gain changes over time, useful for creating natural volume variations.
    
    Parameters
    ----------
    min_gain : float
        Minimum gain in decibels for the envelope.
    max_gain : float
        Maximum gain in decibels for the envelope.
    n_control_points : int, optional
        Number of control points for the gain envelope, by default 10.
        More points create more detailed envelope variations.
    clip : bool, optional
        Whether to clip the output to [-1, 1] range, by default True.
    p : float, optional
        Probability of applying the random gain envelope transform, by default 1.0.
    
    Examples
    --------
    Apply a smooth random gain envelope:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.amplitude import RandomGainEnvelope
    >>> 
    >>> # Create audio samples (1 second at 8kHz)
    >>> samples = np.random.randn(8000) * 0.1
    >>> 
    >>> # Apply smooth gain envelope with 5 control points
    >>> envelope_transform = RandomGainEnvelope(
    ...     min_gain=-12.0, 
    ...     max_gain=6.0, 
    ...     n_control_points=5
    ... )
    >>> result = envelope_transform(samples)
    
    Use in audio processing pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Create dynamic volume processing
    >>> dynamic_pipeline = S.Compose([
    ...     S.RandomGainEnvelope(min_gain=-9.0, max_gain=3.0, n_control_points=8, p=0.7),
    ...     S.Gain(gain=6.0, p=0.5)  # Additional boost
    ... ])
    >>> 
    >>> # Process audio with natural volume variations
    >>> processed = dynamic_pipeline(samples, sample_rate=44100)
    
    Different envelope scenarios:
    
    >>> # Subtle volume variations for music
    >>> subtle_envelope = RandomGainEnvelope(
    ...     min_gain=-3.0, max_gain=3.0, n_control_points=15
    ... )
    >>> 
    >>> # Dramatic variations for sound effects
    >>> dramatic_envelope = RandomGainEnvelope(
    ...     min_gain=-20.0, max_gain=10.0, n_control_points=5
    ... )
    >>> 
    >>> # High-resolution envelope for detailed control
    >>> detailed_envelope = RandomGainEnvelope(
    ...     min_gain=-6.0, max_gain=6.0, n_control_points=50
    ... )
    
    Notes
    -----
    - The envelope is created by linearly interpolating between random gain values
    - More control points create more complex envelope shapes
    - The envelope affects the entire audio sample duration
    - Gain values are converted from dB to linear scale before application
    - The transform preserves audio sample length and format
    
    See Also
    --------
    RandomGain : Apply a single random gain to entire audio
    Gain : Apply a fixed gain to audio samples
    """
    def __init__(self, min_gain: float, max_gain: float, n_control_points: int = 10, clip: bool = True, p: float = 1.0):
        """
        Initialize the RandomGainEnvelope transform.
        
        Parameters
        ----------
        min_gain : float
            Minimum gain in decibels.
        max_gain : float
            Maximum gain in decibels.
        n_control_points : int, optional
            Number of control points for the gain envelope, by default 10.
        clip : bool, optional
            Whether to clip the output to [-1, 1] range, by default True.
        p : float, optional
            Probability of applying the random gain envelope transform, by default 1.0.
        """
        super().__init__(gain=0.0, clip=clip, p=p)
        if min_gain > max_gain:
            raise ValueError("min_gain must be less than or equal to max_gain")
        if n_control_points < 2:
            raise ValueError("n_control_points must be at least 2")
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.n_control_points = n_control_points
    
    def _gain(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply a smoothly varying random gain envelope to the audio samples.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples.
            
        Returns
        -------
        np.ndarray
            Audio samples after applying the random gain envelope and optional clipping.
        """
        n_samples = samples.shape[0]
        control_points = np.linspace(0, n_samples, self.n_control_points)
        random_gains = np.random.uniform(self.min_gain, self.max_gain, self.n_control_points)
        gain_envelope = np.interp(np.arange(n_samples), control_points, random_gains)
        gain_factors = 10 ** (gain_envelope / 20)
        
        result = samples * gain_factors
        if self.clip:
            result = np.clip(result, -1.0, 1.0)
        return result