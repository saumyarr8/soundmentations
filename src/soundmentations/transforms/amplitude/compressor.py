import numpy as np
from soundmentations.core.transforms_interface import BaseTransform

class BaseCompressor(BaseTransform):
    """
    Base class for audio compression transforms.

    This class provides a template for compression transforms that modify
    the amplitude of the input audio sample according to specified parameters.
    Subclasses must implement the _compress method to define the actual
    compression algorithm.
    
    Parameters
    ----------
    threshold : float
        The threshold above which compression is applied, in dB. 
        Typical values range from -40 to -6 dB.
    ratio : float
        The compression ratio to apply. Must be >= 1.0.
        - 1.0 = no compression
        - 2.0 = 2:1 compression (signal 2dB above threshold becomes 1dB above)
        - 10.0 = 10:1 compression (heavy compression)
        - float('inf') = limiting (hard limiting at threshold)
    attack_time : float
        Attack time in milliseconds. How quickly the compressor responds to 
        signals above the threshold. Typical values: 0.1 to 100 ms.
    release_time : float
        Release time in milliseconds. How quickly the compressor stops 
        compressing after the signal falls below threshold.
        Typical values: 10 to 1000 ms.
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.

    Notes
    -----
    Dynamic range compression reduces the volume of loud sounds and can 
    optionally boost quiet sounds, effectively reducing the dynamic range
    of the audio signal. This is commonly used in audio production to
    control peaks and increase the average loudness.

    See Also
    --------
    Compressor : Concrete implementation of a basic compressor
    """
    
    def __init__(self, threshold: float, ratio: float, attack_time: float, release_time: float, p: float = 1.0):
        super().__init__(p)
        self.threshold = threshold
        self.ratio = ratio
        self.attack_time = attack_time
        self.release_time = release_time

    def __call__(self, sample: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply the compression to the audio sample.

        Parameters
        ----------
        sample : np.ndarray
            The audio sample to be compressed.
        sample_rate : int, optional
            Sample rate of the audio, by default 44100.

        Returns
        -------
        np.ndarray
            The compressed audio sample, or the original sample if not applied.
        """
        self.validate_audio(sample, sample_rate)

        if not self.should_apply():
            return sample

        return self._compress(sample, sample_rate)

    def _compress(self, sample: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply compression to the audio sample. Must be implemented by subclasses.

        Parameters
        ----------
        sample : np.ndarray
            The audio sample to be compressed as a 1D numpy array.
        sample_rate : int
            Sample rate of the audio in Hz.

        Returns
        -------
        np.ndarray
            The compressed audio sample with the same shape as input.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _compress method")


class Compressor(BaseCompressor):
    """
    Apply dynamic range compression to the audio sample.

    This compressor uses an envelope follower with configurable attack and 
    release times to track the signal level, then applies gain reduction
    based on a threshold and compression ratio.

    Parameters
    ----------
    threshold : float
        The threshold above which compression is applied, in dB.
        Typical values range from -40 to -6 dB.
    ratio : float
        The compression ratio to apply. Must be >= 1.0.
        - 1.0 = no compression
        - 2.0 = 2:1 compression 
        - 10.0 = 10:1 compression (heavy compression)
    attack_time : float
        Attack time in milliseconds. How quickly the compressor responds to 
        signals above the threshold. Typical values: 0.1 to 100 ms.
    release_time : float
        Release time in milliseconds. How quickly the compressor stops 
        compressing after the signal falls below threshold.
        Typical values: 10 to 1000 ms.
    p : float, optional
        Probability of applying the transform, by default 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from soundmentations.transforms.amplitude.compressor import Compressor
    >>> 
    >>> # Create a compressor with 4:1 ratio and -12dB threshold
    >>> compressor = Compressor(threshold=-12.0, ratio=4.0, 
    ...                        attack_time=5.0, release_time=50.0)
    >>> 
    >>> # Apply to a sine wave
    >>> sample_rate = 44100
    >>> duration = 1.0
    >>> t = np.linspace(0, duration, int(sample_rate * duration))
    >>> audio = np.sin(2 * np.pi * 440 * t) * 0.8  # 440Hz sine wave
    >>> compressed = compressor(audio, sample_rate)

    Notes
    -----
    The compressor implementation uses:
    - Linear threshold conversion from dB
    - Exponential envelope follower with separate attack/release coefficients
    - Logarithmic gain calculation for smooth compression curves

    The envelope follower uses first-order low-pass filtering to smooth
    the absolute value of the input signal, with different time constants
    for attack (signal increasing) and release (signal decreasing).
    """

    def _compress(self, sample: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply compression algorithm to the audio sample.

        This implementation uses an envelope follower to track the signal
        amplitude, then applies gain reduction based on the threshold and ratio.

        Parameters
        ----------
        sample : np.ndarray
            The audio sample to be compressed as a 1D numpy array.
        sample_rate : int
            Sample rate of the audio in Hz.

        Returns
        -------
        np.ndarray
            The compressed audio sample with the same shape as input.

        Notes
        -----
        The compression algorithm:
        1. Converts threshold from dB to linear scale
        2. Calculates attack/release coefficients from time constants
        3. Uses envelope follower to track signal amplitude
        4. Applies gain reduction above the threshold based on the ratio
        """
        # Convert threshold from dB to linear
        threshold_lin = 10 ** (self.threshold / 20.0)
        # Envelope follower parameters
        attack_coeff = np.exp(-1.0 / (0.001 * self.attack_time * sample_rate))
        release_coeff = np.exp(-1.0 / (0.001 * self.release_time * sample_rate))

        envelope = np.zeros_like(sample)
        gain = np.ones_like(sample)
        prev_env = 0.0

        for i, x in enumerate(np.abs(sample)):
            if x > prev_env:
                env = attack_coeff * prev_env + (1 - attack_coeff) * x
            else:
                env = release_coeff * prev_env + (1 - release_coeff) * x
            envelope[i] = env
            prev_env = env

        # Calculate gain reduction
        for i, env in enumerate(envelope):
            if env > threshold_lin:
                # dB above threshold
                db_above = 20 * np.log10(env / threshold_lin)
                db_gain = db_above * (1 - 1 / self.ratio)
                gain[i] = 10 ** (-db_gain / 20.0)
            else:
                gain[i] = 1.0

        return sample * gain
    