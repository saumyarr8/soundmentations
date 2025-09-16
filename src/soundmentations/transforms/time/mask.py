import numpy as np
from soundmentations.core.transforms_interface import BaseTransform


class BaseMask(BaseTransform):
    """
    Base class for masking audio data.

    This class provides a template for masking transforms that modify
    a portion of the input audio sample according to a specified ratio.
    Subclasses must implement the `_mask` method to define the specific
    masking behavior.

    Parameters
    ----------
    mask_ratio : float, optional
        The ratio of audio to mask (0.0 to 1.0, inclusive), by default 0.2.
        For example, 0.2 means 20% of the audio length will be masked.
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.

    Raises
    ------
    ValueError
        If mask_ratio is not between 0.0 and 1.0 (inclusive).

    Notes
    -----
    Masking transforms are commonly used in audio data augmentation
    to simulate various types of audio degradation or to create
    training data that is more robust to missing information.

    See Also
    --------
    TimeMask : Concrete implementation that masks contiguous time segments
    """
    
    def __init__(self, mask_ratio: float = 0.2, p: float = 1.0):
        """
        Initialize the BaseMask transform.

        Parameters
        ----------
        mask_ratio : float, optional
            The ratio of audio to mask (0.0 to 1.0, inclusive), by default 0.2.
        p : float, optional
            Probability of applying the transform, by default 1.0.

        Raises
        ------
        ValueError
            If mask_ratio is not between 0.0 and 1.0 (inclusive).
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0.0 and 1.0 (inclusive).")
        super().__init__(p)
        self.mask_ratio = mask_ratio

    def __call__(self, sample: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply the mask to the audio sample.

        Parameters
        ----------
        sample : np.ndarray
            The audio sample to be masked. Should be a 1D numpy array.
        sample_rate : int, optional
            Sample rate of the audio in Hz, by default 44100.

        Returns
        -------
        np.ndarray
            The masked audio sample with the same shape as input,
            or the original sample if the transform is not applied.
        """
        self.validate_audio(sample, sample_rate)

        if not self.should_apply():
            return sample

        return self._mask(sample, sample_rate)
    
    def _mask(self, sample: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Mask a portion of the audio sample.

        This method must be implemented by subclasses to define the specific
        masking algorithm.

        Parameters
        ----------
        sample : np.ndarray
            The audio sample to be masked as a 1D numpy array.
        sample_rate : int
            Sample rate of the audio in Hz.

        Returns
        -------
        np.ndarray
            The masked audio sample with the same shape as input.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the _mask method.")
    

class Mask(BaseMask):
    """
    Mask a random contiguous segment of audio data with zeros.

    This transform randomly selects a contiguous time segment of the audio
    and replaces it with silence (zeros), simulating audio dropouts,
    temporal masking effects, or packet loss in streaming audio.

    Parameters
    ----------
    mask_ratio : float, optional
        The ratio of audio length to mask (0.0 to 1.0), by default 0.2.
        For example, 0.2 means 20% of the audio duration will be masked.
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from soundmentations.transforms.time.mask import TimeMask
    >>> 
    >>> # Create audio signal (1 second at 44.1kHz)
    >>> sample_rate = 44100
    >>> duration = 1.0
    >>> t = np.linspace(0, duration, int(sample_rate * duration))
    >>> audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    >>> 
    >>> # Create TimeMask that masks 10% of the audio
    >>> time_mask = TimeMask(mask_ratio=0.1, p=1.0)
    >>> masked_audio = time_mask(audio, sample_rate=44100)
    >>> 
    >>> # Verify that some portion is masked
    >>> assert len(masked_audio) == len(audio)
    >>> assert np.sum(masked_audio == 0) > 0  # Some samples are zero
    >>>
    >>> # Example with probability
    >>> probabilistic_mask = TimeMask(mask_ratio=0.2, p=0.5)
    >>> maybe_masked = probabilistic_mask(audio, sample_rate=44100)

    Notes
    -----
    The masking process:
    1. Calculates the number of samples to mask based on mask_ratio
    2. Randomly selects a starting position for the mask
    3. Replaces the selected segment with zeros
    4. Concatenates the unmasked portions with the masked segment

    This transform is useful for:
    - Simulating audio dropouts or glitches
    - Creating training data robust to missing temporal information
    - Augmenting datasets for speech recognition tasks
    - Testing model robustness to temporal discontinuities

    The mask location is uniformly random across the audio sample,
    ensuring no bias toward beginning or end of the audio.
    """
    
    def _mask(self, sample: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Mask a random contiguous segment of the audio sample with zeros.

        The method calculates the mask length based on the mask_ratio,
        randomly selects a starting position, and replaces that segment
        with zeros while preserving the original audio length.

        Parameters
        ----------
        sample : np.ndarray
            The audio sample to be masked as a 1D numpy array.
        sample_rate : int
            Sample rate of the audio in Hz (used for consistency with interface).

        Returns
        -------
        np.ndarray
            The masked audio sample with the same length as input.
            The masked segment is replaced with zeros.

        Notes
        -----
        Edge cases handled:
        - If mask_length <= 0: returns original sample unchanged
        - If mask_length >= sample length: returns array of all zeros
        - Ensures the mask fits entirely within the sample bounds
        """
        mask_length = int(len(sample) * self.mask_ratio)
        
        # Handle edge cases
        if mask_length <= 0:
            return sample.copy()
        if mask_length >= len(sample):
            return np.zeros_like(sample)
        
        # Select random start position ensuring mask fits within bounds
        max_start = len(sample) - mask_length
        start = np.random.randint(0, max_start + 1)
        end = start + mask_length
        
        masked_sample = sample.copy()
        masked_sample[start:end] = 0
        
        return masked_sample