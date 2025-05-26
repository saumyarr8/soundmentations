import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import os
from scipy import signal

def load_audio(file_path: str, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the audio data as a mono numpy array.

    Parameters:
    - file_path (str): Path to the audio file.
    - sample_rate (int, optional): Desired sample rate. If None, uses the original sample rate.

    Returns:
    - Tuple[np.ndarray, int]: Mono audio data as numpy array and sample rate.

    Raises:
    - FileNotFoundError: If the audio file doesn't exist.
    - ValueError: If the audio file format is unsupported.
    - RuntimeError: If resampling fails.
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        audio_data, sr = sf.read(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")

    # Convert to mono if audio has multiple channels
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if needed using scipy
    if sample_rate is not None and sr != sample_rate:
        try:
            # Calculate the number of samples in the resampled audio
            num_samples = int(len(audio_data) * sample_rate / sr)
            audio_data = signal.resample(audio_data, num_samples)
            sr = sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to resample audio from {sr}Hz to {sample_rate}Hz: {str(e)}")
    
    return audio_data, sr