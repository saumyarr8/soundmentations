import numpy as np
from typing import Optional, Union, Tuple
import random


class BaseTrim:
    """
    Base class for audio trimming transforms.
    
    This class provides common functionality for trimming operations including
    parameter validation and probability handling. All trimming subclasses
    inherit from this base class and implement the _trim method.
    """
    
    def __init__(self, p: float = 1.0):
        """
        Initialize the base trimming transform.
        
        Parameters
        ----------
        p : float, optional
            Probability of applying the transform, by default 1.0.
            Must be between 0.0 and 1.0.
        
        Raises
        ------
        TypeError
            If p is not a float or int.
        ValueError
            If p is not between 0.0 and 1.0.
        """
        # Validate probability p
        if not isinstance(p, (float, int)):
            raise TypeError("p must be a float or an integer")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0.0 and 1.0")
        self.p = p
    
    def __call__(self, samples: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply the trimming transform to the audio samples with probability p.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples as a 1D numpy array (mono audio only).
        sample_rate : int, optional
            Sample rate of the audio, by default 44100.
        
        Returns
        -------
        np.ndarray
            Trimmed audio samples (or original if probability check fails).
        
        Raises
        ------
        TypeError
            If samples is not a numpy array or sample_rate is not an integer.
        ValueError
            If samples is empty, not 1D, or sample_rate is not positive.
        """
        # Validate input samples
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy array")
        if samples.size == 0:
            raise ValueError("Input samples cannot be empty")
        if samples.ndim != 1:
            raise ValueError("samples must be a 1D array (mono audio only)")
        
        # Validate sample rate
        if not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        # Apply probability check - skip transformation if random value exceeds p
        if random.random() > self.p:
            return samples

        # Apply the specific trimming logic implemented by subclass
        return self._trim(samples, sample_rate)
    
    def _trim(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Abstract method to be implemented by subclasses for specific trimming logic.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (validated by __call__).
        sample_rate : int
            Sample rate of the audio (validated by __call__).
        
        Returns
        -------
        np.ndarray
            Trimmed audio samples.
        
        Notes
        -----
        This method is called only after input validation and probability check.
        Subclasses must implement this method with their specific trimming strategy.
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the _trim method")


class Trim(BaseTrim):
    """
    Trim audio to keep only the portion between start_time and end_time.
    
    This is the most basic trimming operation that allows specifying exact
    start and end times for the audio segment to keep.
    
    Parameters
    ----------
    start_time : float, optional
        Start time in seconds to begin keeping audio, by default 0.0.
        Must be non-negative.
    end_time : float, optional
        End time in seconds to stop keeping audio, by default None.
        If None, keeps audio until the end. Must be greater than start_time.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Trim audio to specific time range:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import Trim
    >>> 
    >>> # Create 5 seconds of audio at 44.1kHz
    >>> audio = np.random.randn(220500)
    >>> 
    >>> # Keep audio from 1.5 to 3.0 seconds
    >>> trim_transform = Trim(start_time=1.5, end_time=3.0)
    >>> trimmed = trim_transform(audio, sample_rate=44100)
    >>> print(len(trimmed) / 44100)  # 1.5 seconds
    
    Use in a pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Extract middle portion and apply gain
    >>> pipeline = S.Compose([
    ...     S.Trim(start_time=1.0, end_time=4.0, p=1.0),
    ...     S.Gain(gain=6.0, p=0.5)
    ... ])
    >>> 
    >>> result = pipeline(audio, sample_rate=44100)
    """
    
    def __init__(self, start_time: float = 0.0, end_time: Optional[float] = None, p: float = 1.0):
        """
        Initialize the trim transform with specific start and end times.
        
        Parameters
        ----------
        start_time : float, optional
            Start time in seconds, by default 0.0. Must be non-negative.
        end_time : float, optional
            End time in seconds, by default None. Must be greater than start_time.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        
        Raises
        ------
        TypeError
            If start_time or end_time is not a number.
        ValueError
            If times are invalid (negative start, end <= start).
        """
        super().__init__(p)
        
        # Validate start_time
        if not isinstance(start_time, (float, int)):
            raise TypeError("start_time must be a number")
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        
        # Validate end_time if provided
        if end_time is not None:
            if not isinstance(end_time, (float, int)):
                raise TypeError("end_time must be a number")
            if end_time <= start_time:
                raise ValueError("end_time must be greater than start_time")
        
        self.start_time = start_time
        self.end_time = end_time

    def _trim(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply trimming to keep audio between start_time and end_time.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (mono audio only).
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Trimmed audio samples between specified times.
        
        Raises
        ------
        ValueError
            If time bounds are invalid or exceed audio duration.
        """
        # Calculate audio duration in seconds
        audio_duration = len(samples) / sample_rate
        
        # Validate start_time against audio duration
        if self.start_time >= audio_duration:
            raise ValueError(f"start_time ({self.start_time}s) exceeds audio duration ({audio_duration:.2f}s)")
        
        # Calculate start index in samples
        start_idx = int(self.start_time * sample_rate)
        
        # Handle end_time: use end of audio if None, otherwise validate and convert
        if self.end_time is None:
            end_idx = len(samples)
        else:
            if self.end_time > audio_duration:
                raise ValueError(f"end_time ({self.end_time}s) exceeds audio duration ({audio_duration:.2f}s)")
            end_idx = int(self.end_time * sample_rate)
        
        # Final validation: ensure we have some audio left after trimming
        if start_idx >= end_idx:
            raise ValueError("No audio remains after trimming")
            
        # Return the trimmed segment
        return samples[start_idx:end_idx]


class RandomTrim(BaseTrim):
    """
    Randomly trim audio by selecting a random segment of specified duration.
    
    This transform randomly selects a continuous segment from the audio,
    useful for data augmentation where you want random crops of fixed or
    variable duration.
    
    Parameters
    ----------
    duration : float or tuple of float
        If float, exact duration to keep in seconds.
        If tuple (min_duration, max_duration), random duration in range.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Fixed duration random trimming:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import RandomTrim
    >>> 
    >>> # Always keep 2 seconds randomly
    >>> trim_transform = RandomTrim(duration=2.0)
    >>> trimmed = trim_transform(audio, sample_rate=44100)
    >>> print(len(trimmed) / 44100)  # 2.0 seconds
    
    Variable duration random trimming:
    
    >>> # Keep 1-3 seconds randomly
    >>> variable_trim = RandomTrim(duration=(1.0, 3.0))
    >>> result = variable_trim(audio, sample_rate=44100)
    
    Use for data augmentation:
    
    >>> import soundmentations as S
    >>> 
    >>> # Random crop and normalize for training
    >>> augment = S.Compose([
    ...     S.RandomTrim(duration=(0.5, 2.5), p=0.8),
    ...     S.PadToLength(pad_length=88200, p=1.0),  # 2 seconds
    ...     S.Gain(gain=(-6, 6), p=0.5)
    ... ])
    >>> 
    >>> augmented = augment(training_audio, sample_rate=44100)
    """
    
    def __init__(self, duration: Union[float, Tuple[float, float]], p: float = 1.0):
        """
        Initialize the random trim transform with duration specification.
        
        Parameters
        ----------
        duration : float or tuple of float
            Either a single duration or range of durations in seconds.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        
        Raises
        ------
        TypeError
            If duration is not a number or tuple.
        ValueError
            If duration values are invalid.
        """
        super().__init__(p)
        
        if isinstance(duration, (int, float)):
            # Single duration value
            if duration <= 0:
                raise ValueError("duration must be positive")
            self.min_duration = duration
            self.max_duration = duration
            
        elif isinstance(duration, (tuple, list)) and len(duration) == 2:
            # Duration range
            min_dur, max_dur = duration
            
            # Validate both values are numbers
            if not isinstance(min_dur, (float, int)) or not isinstance(max_dur, (float, int)):
                raise TypeError("duration values must be numbers")
            
            # Validate values are positive
            if min_dur <= 0 or max_dur <= 0:
                raise ValueError("duration values must be positive")
            
            # Validate range is valid
            if min_dur > max_dur:
                raise ValueError("min_duration must be <= max_duration")
                
            self.min_duration = min_dur
            self.max_duration = max_dur
        else:
            raise ValueError("duration must be a float or tuple of (min_duration, max_duration)")

    def _trim(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply random trimming to keep a random segment of the specified duration.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (mono audio only).
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Randomly trimmed audio samples.
        
        Raises
        ------
        ValueError
            If duration exceeds audio length.
        """
        # Calculate audio duration in seconds
        audio_duration = len(samples) / sample_rate
        
        # Choose random duration within specified range
        target_duration = random.uniform(self.min_duration, self.max_duration)
        
        # Validate target duration doesn't exceed audio length
        if target_duration >= audio_duration:
            raise ValueError(f"target_duration ({target_duration:.2f}s) exceeds audio duration ({audio_duration:.2f}s)")
        
        # Calculate maximum possible start time to fit the target duration
        max_start_time = audio_duration - target_duration
        
        # Choose random start position
        start_time = random.uniform(0, max_start_time)
        end_time = start_time + target_duration
        
        # Convert times to sample indices
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        # Safety check: ensure we don't exceed array bounds
        end_idx = min(end_idx, len(samples))
        
        return samples[start_idx:end_idx]


class StartTrim(BaseTrim):
    """
    Trim audio to keep only the portion starting from start_time to the end.
    
    This removes the beginning of the audio up to start_time, keeping
    everything after that point.
    
    Parameters
    ----------
    start_time : float, optional
        Start time in seconds to begin keeping audio, by default 0.0.
        Must be non-negative.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Remove silence from beginning:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import StartTrim
    >>> 
    >>> # Remove first 2 seconds
    >>> trim_transform = StartTrim(start_time=2.0)
    >>> trimmed = trim_transform(audio, sample_rate=44100)
    
    Use in preprocessing pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Remove intro and normalize
    >>> preprocess = S.Compose([
    ...     S.StartTrim(start_time=1.5, p=1.0),
    ...     S.PadToLength(pad_length=132300, p=1.0)  # 3 seconds
    ... ])
    >>> 
    >>> processed = preprocess(raw_audio, sample_rate=44100)
    """
    
    def __init__(self, start_time: float = 0.0, p: float = 1.0):
        """
        Initialize the start trim transform.
        
        Parameters
        ----------
        start_time : float, optional
            Time in seconds from which to start keeping audio, by default 0.0.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        
        Raises
        ------
        TypeError
            If start_time is not a number.
        ValueError
            If start_time is negative.
        """
        super().__init__(p)
        
        # Validate start_time
        if not isinstance(start_time, (float, int)):
            raise TypeError("start_time must be a number")
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        
        self.start_time = start_time

    def _trim(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply trimming to keep audio from start_time to the end.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (mono audio only).
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Audio samples starting from start_time.
        
        Raises
        ------
        ValueError
            If start_time exceeds audio duration.
        """
        # Calculate audio duration in seconds
        audio_duration = len(samples) / sample_rate
        
        # Validate start_time doesn't exceed audio duration
        if self.start_time >= audio_duration:
            raise ValueError(f"start_time ({self.start_time}s) exceeds audio duration ({audio_duration:.2f}s)")
        
        # Convert start time to sample index
        start_idx = int(self.start_time * sample_rate)
        
        # Return everything from start_idx to the end
        return samples[start_idx:]


class EndTrim(BaseTrim):
    """
    Trim audio to keep only the portion from the start to end_time.
    
    This removes the end of the audio after end_time, keeping everything
    before that point.
    
    Parameters
    ----------
    end_time : float
        End time in seconds to stop keeping audio. Must be positive.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Keep only first part of audio:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import EndTrim
    >>> 
    >>> # Keep first 5 seconds only
    >>> trim_transform = EndTrim(end_time=5.0)
    >>> trimmed = trim_transform(audio, sample_rate=44100)
    
    Use for consistent audio lengths:
    
    >>> import soundmentations as S
    >>> 
    >>> # Ensure maximum 10 seconds
    >>> limit_length = S.Compose([
    ...     S.EndTrim(end_time=10.0, p=1.0),
    ...     S.Gain(gain=3.0, p=0.3)
    ... ])
    >>> 
    >>> limited = limit_length(long_audio, sample_rate=44100)
    """
    
    def __init__(self, end_time: float, p: float = 1.0):
        """
        Initialize the end trim transform.
        
        Parameters
        ----------
        end_time : float
            Time in seconds at which to stop keeping audio.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        
        Raises
        ------
        TypeError
            If end_time is not a number.
        ValueError
            If end_time is not positive.
        """
        super().__init__(p)
        
        # Validate end_time
        if not isinstance(end_time, (float, int)):
            raise TypeError("end_time must be a number")
        if end_time <= 0:
            raise ValueError("end_time must be positive")
        
        self.end_time = end_time

    def _trim(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply trimming to keep audio from start to end_time.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (mono audio only).
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Audio samples up to end_time.
        
        Notes
        -----
        If end_time exceeds audio duration, returns the full audio unchanged.
        """
        # Calculate audio duration in seconds
        audio_duration = len(samples) / sample_rate
        
        # If end_time exceeds audio duration, return all audio
        if self.end_time >= audio_duration:
            return samples
        
        # Convert end time to sample index
        end_idx = int(self.end_time * sample_rate)
        
        # Return everything from start to end_idx
        return samples[:end_idx]


class CenterTrim(BaseTrim):
    """
    Trim audio to keep only the center portion of specified duration.
    
    This extracts a segment from the middle of the audio, useful for
    focusing on the main content while removing silence at the beginning
    and end.
    
    Parameters
    ----------
    duration : float
        Duration of the center portion to keep in seconds. Must be positive.
    p : float, optional
        Probability of applying the transform, by default 1.0.
    
    Examples
    --------
    Extract center content:
    
    >>> import numpy as np
    >>> from soundmentations.transforms.time import CenterTrim
    >>> 
    >>> # Keep 3 seconds from center
    >>> trim_transform = CenterTrim(duration=3.0)
    >>> trimmed = trim_transform(audio, sample_rate=44100)
    >>> print(len(trimmed) / 44100)  # 3.0 seconds
    
    Use for focusing on main content:
    
    >>> import soundmentations as S
    >>> 
    >>> # Extract center and enhance
    >>> focus_pipeline = S.Compose([
    ...     S.CenterTrim(duration=4.0, p=1.0),
    ...     S.Gain(gain=6.0, p=0.6),
    ...     S.PadToLength(pad_length=176400, p=1.0)  # 4 seconds
    ... ])
    >>> 
    >>> focused = focus_pipeline(noisy_audio, sample_rate=44100)
    """
    
    def __init__(self, duration: float, p: float = 1.0):
        """
        Initialize the center trim transform.
        
        Parameters
        ----------
        duration : float
            Duration in seconds of the center segment to keep.
        p : float, optional
            Probability of applying the transform, by default 1.0.
        
        Raises
        ------
        TypeError
            If duration is not a number.
        ValueError
            If duration is not positive.
        """
        super().__init__(p)
        
        # Validate duration
        if not isinstance(duration, (float, int)):
            raise TypeError("duration must be a number")
        if duration <= 0:
            raise ValueError("duration must be positive")
            
        self.duration = duration

    def _trim(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply trimming to keep the center portion of the specified duration.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples (mono audio only).
        sample_rate : int
            Sample rate of the audio.
        
        Returns
        -------
        np.ndarray
            Center segment of the audio.
        
        Raises
        ------
        ValueError
            If duration exceeds audio length.
        """
        # Calculate audio duration in seconds
        audio_duration = len(samples) / sample_rate
        
        # Validate duration doesn't exceed audio length
        if self.duration >= audio_duration:
            raise ValueError(f"duration ({self.duration:.2f}s) exceeds audio duration ({audio_duration:.2f}s)")
        
        # Calculate start and end times for the center portion
        # Center the segment by placing equal amounts of audio on both sides
        start_time = (audio_duration - self.duration) / 2
        end_time = start_time + self.duration
        
        # Convert times to sample indices
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        # Ensure we don't exceed array bounds
        end_idx = min(end_idx, len(samples))
        
        return samples[start_idx:end_idx]