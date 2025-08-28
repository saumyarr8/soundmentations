import random

__all__ = [
    "BaseCompose",
    "Compose",
    "OneOf",
]

class BaseCompose:
    """
    Base class for composing multiple transforms into a sequential pipeline.
    
    This class provides the fundamental functionality for chaining transforms
    together, where each transform is applied sequentially to the audio data.
    
    Parameters
    ----------
    transforms : list
        List of transform objects to apply sequentially.
        Each transform must have a __call__ method that accepts
        (samples, sample_rate) parameters.
    
    Notes
    -----
    This is an internal base class. Use the Compose class instead.
    """
    
    def __init__(self, transforms):
        """
        Initialize the base composition with a list of transforms.
        
        Parameters
        ----------
        transforms : list
            List of transform instances to chain together.
        """
        self.transforms = transforms

    def __call__(self, samples, sample_rate=44100):
        """
        Apply all transforms sequentially to the input audio.
        
        Each transform in the pipeline receives the output of the previous
        transform, creating a chain of audio processing operations.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples as a 1D numpy array (mono audio).
        sample_rate : int, optional
            Sample rate of the audio, by default 44100.
        
        Returns
        -------
        np.ndarray
            Audio samples after applying all transforms in sequence.
        """
        # Apply each transform sequentially
        # Output of one transform becomes input to the next
        for t in self.transforms:
            samples = t(samples, sample_rate)
        return samples
    

class Compose(BaseCompose):
    """
    Compose multiple audio transforms into a sequential pipeline.
    
    This class allows you to chain multiple transforms together into a single
    callable object. Transforms are applied in the order they appear in the list,
    with each transform receiving the output of the previous one.
    
    Parameters
    ----------
    transforms : list
        List of transform objects to apply sequentially.
        Each transform must implement __call__(samples, sample_rate).
    
    Examples
    --------
    Create a basic augmentation pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Define individual transforms
    >>> pipeline = S.Compose([
    ...     S.RandomTrim(duration=(1.0, 3.0), p=0.8),
    ...     S.Pad(pad_length=44100, p=0.6),
    ...     S.Gain(gain=6.0, p=0.5)
    ... ])
    >>> 
    >>> # Apply to audio
    >>> augmented = pipeline(audio_samples, sample_rate=44100)
    
    Complex preprocessing pipeline:
    
    >>> # ML training data preparation
    >>> ml_pipeline = S.Compose([
    ...     S.CenterTrim(duration=2.0),              # Extract 2s from center
    ...     S.PadToLength(pad_length=88200),         # Normalize to exactly 2s
    ...     S.Gain(gain=3.0, p=0.7),                # Boost volume 70% of time
    ...     S.FadeIn(duration=0.1, p=0.5),          # Smooth start 50% of time
    ...     S.FadeOut(duration=0.1, p=0.5)          # Smooth end 50% of time
    ... ])
    >>> 
    >>> # Process batch of audio files
    >>> for audio in audio_batch:
    ...     processed = ml_pipeline(audio, sample_rate=16000)
    
    Audio enhancement pipeline:
    
    >>> # Clean up audio recordings
    >>> enhance_pipeline = S.Compose([
    ...     S.StartTrim(start_time=0.5),            # Remove first 0.5s
    ...     S.EndTrim(end_time=10.0),               # Keep max 10s
    ...     S.Gain(gain=6.0),                       # Boost volume
    ...     S.FadeIn(duration=0.2),                 # Smooth fade-in
    ...     S.FadeOut(duration=0.2)                 # Smooth fade-out
    ... ])
    >>> 
    >>> enhanced = enhance_pipeline(noisy_audio, sample_rate=44100)
    
    Notes
    -----
    - Transforms are applied in order: first transform in list is applied first
    - Each transform receives the output of the previous transform
    - Probability parameters (p) in individual transforms are respected
    - The pipeline preserves mono audio format throughout
    - All transforms must accept (samples, sample_rate) parameters
    
    See Also
    --------
    Individual transforms : Gain, Trim, Pad, RandomTrim, FadeIn, FadeOut
    """
    
    def __init__(self, transforms):
        """
        Initialize the composition pipeline.
        
        Parameters
        ----------
        transforms : list
            List of transform instances to chain together.
            Each transform must implement __call__(samples, sample_rate).
        
        Raises
        ------
        TypeError
            If transforms is not a list or if any item is not callable.
        ValueError
            If transforms list is empty.
        
        Examples
        --------
        >>> # Simple pipeline
        >>> pipeline = S.Compose([
        ...     S.Trim(start_time=1.0, end_time=3.0),
        ...     S.Gain(gain=6.0)
        ... ])
        >>> 
        >>> # Empty pipeline (not recommended)
        >>> # This will raise ValueError
        >>> # pipeline = S.Compose([])
        """
        if not isinstance(transforms, list):
            raise TypeError("transforms must be a list")
        if len(transforms) == 0:
            raise ValueError("transforms list cannot be empty")
        
        # Validate that all transforms are callable
        for i, transform in enumerate(transforms):
            if not callable(transform):
                raise TypeError(f"Transform at index {i} is not callable")
        
        # Initialize the base class
        super().__init__(transforms)
    
    def __repr__(self):
        """
        Return string representation of the composition.
        
        Returns
        -------
        str
            String showing the composed transforms.
        """
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose([{', '.join(transform_names)}])"
    
    def __len__(self):
        """
        Return the number of transforms in the pipeline.
        
        Returns
        -------
        int
            Number of transforms in the composition.
        """
        return len(self.transforms)
    
    def __getitem__(self, index):
        """
        Get a transform by index.
        
        Parameters
        ----------
        index : int
            Index of the transform to retrieve.
        
        Returns
        -------
        Transform
            The transform at the specified index.
        """
        return self.transforms[index]
    
class OneOf(BaseCompose):
    """
    Apply one transform from a list of transforms, chosen at random.
    
    This class allows you to randomly select and apply one transform from
    a provided list. Each transform has an equal chance of being selected
    unless specified otherwise.
    
    Parameters
    ----------
    transforms : list
        List of transform objects to choose from.
        Each transform must implement __call__(samples, sample_rate).
    p : float, optional
        Probability of applying one of the transforms, by default 1.0.
        If not applied, the input audio is returned unchanged.
    
    Examples
    --------
    Create a random augmentation pipeline:
    
    >>> import soundmentations as S
    >>> 
    >>> # Define individual transforms
    >>> pipeline = S.OneOf([
    ...     S.Gain(gain=6.0),
    ...     S.FadeIn(duration=0.5),
    ...     S.FadeOut(duration=0.5)
    ... ], p=0.9)
    >>> 
    >>> # Apply to audio
    >>> augmented = pipeline(audio_samples, sample_rate=44100)
    
    Randomized preprocessing pipeline:
    
    >>> # ML training data preparation with randomness
    >>> ml_pipeline = S.OneOf([
    ...     S.RandomTrim(duration=(1.0, 3.0)),
    ...     S.Pad(pad_length=44100),
    ...     S.Gain(gain=3.0)
    ... ], p=0.8)
    >>> 
    >>> # Process batch of audio files
    >>> for audio in audio_batch:
    ...     processed = ml_pipeline(audio, sample_rate=16000)
    
    Audio enhancement with random effects:
    
    >>> # Clean up audio recordings with variability
    >>> enhance_pipeline = S.OneOf([
    ...     S.StartTrim(start_time=0.5),
    ...     S.EndTrim(end_time=10.0),
    ...     S.FadeIn(duration=0.2),
    ...     S.FadeOut(duration=0.2)
    ... ], p=0.7)
    >>> 
    >>> enhanced = enhance_pipeline(noisy_audio, sample_rate=44100)
    
    Notes
    -----
    - Only one transform is applied per call, chosen at random
    - Probability parameter (p) controls likelihood of applying any transform
      - If not applied, input audio is returned unchanged
    """
    
    def __init__(self, transforms, p=1.0):
        """
        Initialize the OneOf composition.
        
        Parameters
        ----------
        transforms : list
            List of transform instances to choose from.
            Each transform must implement __call__(samples, sample_rate).
        p : float, optional
            Probability of applying one of the transforms, by default 1.0.
        
        Raises
        ------
        TypeError
            If transforms is not a list or if any item is not callable.
        ValueError
            If transforms list is empty or if p is not between 0 and 1.
        
        Examples
        --------
        >>> # Simple random pipeline
        >>> pipeline = S.OneOf([
        ...     S.Gain(gain=6.0),
        ...     S.FadeIn(duration=0.5)
        ... ], p=0.9)
        """
        if not isinstance(transforms, list):
            raise TypeError("transforms must be a list")
        if len(transforms) == 0:
            raise ValueError("transforms list cannot be empty")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        
        for i, transform in enumerate(transforms):
            if not callable(transform):
                raise TypeError(f"Transform at index {i} is not callable")
        
        super().__init__(transforms)
        self.p = p
    
    def __call__(self, samples, sample_rate=44100):
        """
        Apply one randomly selected transform to the input audio.
        
        Randomly selects one transform from the list and applies it
        to the audio samples with probability p.
        
        Parameters
        ----------
        samples : np.ndarray
            Input audio samples as a 1D numpy array (mono audio).
        sample_rate : int, optional
            Sample rate of the audio, by default 44100.
        
        Returns
        -------
        np.ndarray
            Audio samples after applying the randomly selected transform,
            or original samples if probability check fails.
        """

        if random.random() > self.p:
            return samples
        
        selected_transform = random.choice(self.transforms)
        return selected_transform(samples, sample_rate)
    
    def __repr__(self):
        """
        Return string representation of the OneOf composition.
        
        Returns
        -------
        str
            String showing the OneOf transforms and probability.
        """
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"OneOf([{', '.join(transform_names)}], p={self.p})"
    
    def __len__(self):
        """
        Return the number of transforms in the OneOf composition.
        
        Returns
        -------
        int
            Number of transforms available for selection.
        """
        return len(self.transforms)
    
    def __getitem__(self, index):
        """
        Get a transform by index.
        
        Parameters
        ----------
        index : int
            Index of the transform to retrieve.
        
        Returns
        -------
        Transform
            The transform at the specified index.
        """
        return self.transforms[index]