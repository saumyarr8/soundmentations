import pytest
import numpy as np
import tempfile
import os
import soundfile as sf

from soundmentations import load_audio

class TestLoadAudio:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up after each test method."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def create_test_audio_file(self, filename: str, sample_rate: int = 44100, 
                              duration: float = 1.0, channels: int = 1) -> str:
        """Helper method to create a test audio file."""
        file_path = os.path.join(self.temp_dir, filename)
        
        # Generate test audio data (sine wave)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        if channels > 1:
            audio_data = np.column_stack([audio_data] * channels)
        
        sf.write(file_path, audio_data, sample_rate)
        return file_path
    
    def test_load_audio_mono_file(self):
        file_path = self.create_test_audio_file("test_mono.wav", 44100, 1.0, 1)
        audio_data, sr = load_audio(file_path)
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.ndim == 1  # mono
        assert sr == 44100
        assert len(audio_data) == 44100
    
    def test_load_audio_stereo_file(self):
        file_path = self.create_test_audio_file("test_stereo.wav", 44100, 1.0, 2)
        audio_data, sr = load_audio(file_path)
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.ndim == 1  # converted to mono
        assert sr == 44100
        assert len(audio_data) == 44100
    
    def test_load_audio_file_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            load_audio("/non/existent/path.wav")
        assert "Audio file not found" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__])
