# Soundmentations

A Python library for audio data augmentation and transformations, inspired by Albumentations but designed specifically for audio processing.

## ⚠️ Important Note

**Soundmentations currently supports mono audio only.** Any multichannel audio will be automatically converted to mono by taking the mean of all channels during loading. This ensures consistent processing across all transforms.

**Coming Soon:**
- 🎵 Additional audio transforms (frequency, effects, noise)
- 📦 Bounding box support for audio annotations
- 🔀 Multichannel audio support
- 🎛️ Advanced spectral transformations

## Features

- 🎵 **Time-based transforms**: Trim, pad, and manipulate audio timing
- 🔊 **Amplitude transforms**: Control volume, gain, and dynamic range
- 🎲 **Probabilistic augmentation**: Apply transforms with configurable probability
- 🔗 **Chainable pipeline**: Compose multiple transforms together
- 📊 **NumPy compatible**: Works seamlessly with numpy arrays
- 🎯 **Mono audio focused**: Optimized for single-channel audio processing

## Installation

```bash
pip install soundmentations
```

Or install from source:

```bash
git clone https://github.com/saumyarr8/soundmentations.git
cd soundmentations
pip install -e .
```

## Quick Start

```python
import numpy as np
from soundmentations import Compose, Trim, RandomTrim, Pad, Gain

# Load your audio data (as numpy array)
audio_samples = np.random.randn(44100)  # 1 second of audio at 44.1kHz
sample_rate = 44100

# Create an augmentation pipeline
augment = Compose([
    RandomTrim(duration=(0.5, 2.0), p=0.8),  # Random trim with 80% probability
    Pad(pad_length=22050, p=0.5),            # Pad to 0.5 seconds with 50% probability
    Gain(gain=(-3, 3), p=0.7)                # Random gain between -3dB and +3dB
])

# Apply augmentations to audio
augmented_samples = augment(samples=audio_samples, sample_rate=sample_rate)
```

## Audio Loading & Mono Conversion

```python
from soundmentations.utils.audio import load_audio

# Load audio file - automatically converted to mono if multichannel
samples, sample_rate = load_audio("path/to/stereo_audio.wav")
print(samples.shape)  # (n_samples,) - always 1D mono array

# For stereo input: [left_channel, right_channel] -> mean([left, right])
# For 5.1 surround: [L, R, C, LFE, LS, RS] -> mean([L, R, C, LFE, LS, RS])
```

## Available Transforms

### Time-based Transforms

```python
from soundmentations.transforms.time import (
    Trim, RandomTrim, StartTrim, EndTrim, CenterTrim,
    Pad, CenterPad, StartPad, PadToLength, CenterPadToLength, PadToMultiple
)

# Trimming transforms
Trim(start_time=1.0, end_time=3.0)           # Keep audio between 1-3 seconds
RandomTrim(duration=2.0)                     # Random 2-second segment
RandomTrim(duration=(1.0, 3.0))             # Random segment between 1-3 seconds
StartTrim(start_time=0.5)                    # Remove first 0.5 seconds
EndTrim(end_time=5.0)                        # Keep only first 5 seconds
CenterTrim(duration=3.0)                     # Keep 3 seconds from center

# Padding transforms
Pad(pad_length=44100)                        # Pad to minimum 1 second
CenterPad(pad_length=44100)                  # Center padding
StartPad(pad_length=44100)                   # Pad at beginning
PadToLength(pad_length=44100)                # Exact length (pad or trim)
CenterPadToLength(pad_length=44100)          # Exact length, centered
PadToMultiple(multiple=1024)                 # Pad to multiple of 1024 (STFT-friendly)
```

### Amplitude Transforms

```python
from soundmentations.transforms.amplitude import Gain

# Gain transforms
Gain(gain=6.0)                               # Fixed +6dB gain
Gain(gain=(-12, 12))                         # Random gain between -12dB and +12dB
```

## Transform Parameters

All transforms support the following common parameters:

- `p` (float): Probability of applying the transform (0.0 to 1.0, default 1.0)

```python
# Apply trim with 70% probability
trim = Trim(start_time=1.0, end_time=3.0, p=0.7)
```

## Compose Pipeline

Chain multiple transforms together:

```python
from soundmentations import Compose

# Create a complex augmentation pipeline
augment = Compose([
    RandomTrim(duration=(0.8, 2.5), p=0.8),     # Random crop
    CenterPadToLength(pad_length=44100, p=0.6),  # Normalize to 1 second
    Gain(gain=(-6, 6), p=0.5),                  # Random volume adjustment
])

# Apply to your audio
augmented = augment(samples=audio_data, sample_rate=44100)
```

## Examples

### Data Augmentation for Machine Learning

```python
import numpy as np
from soundmentations import Compose, RandomTrim, Pad, Gain

# Create augmentation pipeline for training data
train_augment = Compose([
    RandomTrim(duration=(1.0, 3.0), p=0.8),    # Variable length crops
    PadToLength(pad_length=48000, p=1.0),       # Normalize length
    Gain(gain=(-10, 10), p=0.6),               # Volume variation
])

# Augment a batch of audio samples
def augment_batch(audio_batch, sample_rate=16000):
    return [train_augment(samples=audio, sample_rate=sample_rate) 
            for audio in audio_batch]
```

### Audio Preprocessing Pipeline

```python
# Preprocessing pipeline for consistent audio format
preprocess = Compose([
    CenterTrim(duration=5.0),                   # Take 5 seconds from center
    PadToLength(pad_length=80000),              # Ensure exactly 5 seconds at 16kHz
])

# Use for inference
processed_audio = preprocess(samples=raw_audio, sample_rate=16000)
```

## Mono Audio Processing Note

**Important:** All transforms expect and return 1D numpy arrays (mono audio). If you need to process multichannel audio:

```python
# Current approach (automatic conversion)
samples, sr = load_audio("stereo_file.wav")  # Returns mono via mean()
augmented = augment(samples, sample_rate=sr)

# Future multichannel support (coming soon)
# samples, sr = load_audio("stereo_file.wav", mono=False)  # Keep channels
# augmented = augment(samples, sample_rate=sr)  # Process each channel
```

## API Reference

### Transform Base Classes

- `BaseTrim`: Base class for all trimming operations
- `BasePad`: Base class for all padding operations  
- `BaseGain`: Base class for all gain operations

### Utilities

- `load_audio(file_path)`: Load audio file as mono numpy array using soundfile and scipy
- `Compose([transforms])`: Chain multiple transforms together

## Requirements

- Python 3.9+
- NumPy
- soundfile (for audio loading)
- scipy

## Roadmap

### v0.2.0 (Coming Soon)
- 🎛️ **Frequency transforms**: Filters, EQ, pitch shifting
- 🎵 **Effect transforms**: Reverb, echo, distortion
- 🔊 **Noise transforms**: Add background noise

### v0.3.0 (Planned)
- 📦 **Bounding box support**: Audio annotation and localization

### v1.0.0 (Future)
- 🔀 **Multichannel support**: Full stereo and surround sound processing
- 🚀 **Performance optimizations**: GPU acceleration and faster processing
- 📊 **Advanced spectral**: Mel-frequency, MFCC, and other spectral transforms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Current focus areas:
- Additional transform implementations
- Performance improvements
- Documentation enhancements
- Test coverage expansion

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Soundmentations in your research, please cite:

```bibtex
@software{soundmentations,
  title={Soundmentations: Audio Data Augmentation Library},
  author={Saumya Ranjan},
  url={https://github.com/saumyarr8/soundmentations},
  year={2025}
}
```

## Changelog

### v0.1.0
- Initial release
- Time-based transforms (trim, pad)
- Amplitude transforms (gain)
- Probabilistic augmentation
- Compose pipeline
- Audio loading utilities
- Mono audio processing

---

**Soundmentations** - Making audio augmentation simple and powerful! 🎵

*Note: This library is actively developed. Star the repo to stay updated on new features and releases.*