import numpy as np
from typing import Optional

class Trim:
    """
    Trim audio to keep only the portion between start_time and end_time.
    
    Parameters:
    - start_time (float): Start time in seconds to begin keeping audio. Default is 0.0.
    - end_time (float, optional): End time in seconds to stop keeping audio. 
                                  If None, keeps audio until the end.
    """
    
    def __init__(self, start_time: float = 0.0, end_time: Optional[float] = None):
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
            
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, samples: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply trimming to keep audio between start_time and end_time.
        
        Parameters:
        - samples (np.ndarray): Input audio samples
        - sample_rate (int): Sample rate of the audio
        
        Returns:
        - np.ndarray: Trimmed audio samples between specified times
        
        Raises:
        - ValueError: If time bounds are invalid or exceed audio duration
        """
        if len(samples) == 0:
            raise ValueError("Input samples cannot be empty")
            
        audio_duration = len(samples) / sample_rate
        
        # Validate start_time
        if self.start_time >= audio_duration:
            raise ValueError(f"start_time ({self.start_time}s) exceeds audio duration ({audio_duration:.2f}s)")
        
        # Calculate indices
        start_idx = int(self.start_time * sample_rate)
        
        if self.end_time is None:
            end_idx = len(samples)
        else:
            if self.end_time > audio_duration:
                raise ValueError(f"end_time ({self.end_time}s) exceeds audio duration ({audio_duration:.2f}s)")
            end_idx = int(self.end_time * sample_rate)
        
        # Ensure we have some audio left
        if start_idx >= end_idx:
            raise ValueError("No audio remains after trimming")
            
        return samples[start_idx:end_idx]