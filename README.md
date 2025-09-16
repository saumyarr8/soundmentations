# Soundmentations

[![Documentation Status](https://github.com/saumyarr8/soundmentations/actions/workflows/deploy-docs.yml/badge.svg)](https://saumyarr8.github.io/soundmentations/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python library for audio data augmentation and transformations, inspired by [Albumentations](https://albumentations.ai/) but designed specifically for audio processing. Perfect for machine learning pipelines, audio preprocessing, and data augmentation workflows.

## Key Features

- **Comprehensive Audio Transforms**: 25+ transforms covering time, amplitude, and frequency domains
- **Probabilistic Augmentation**: Apply transforms with configurable probability for robust data augmentation
- **Composable Pipelines**: Chain multiple transforms together with `Compose` and `OneOf`
- **NumPy Compatible**: Seamless integration with numpy arrays and scientific Python ecosystem
- **Production Ready**: Extensively tested, documented, and optimized for real-world usage
- **Easy to Use**: Simple, intuitive API inspired by successful augmentation libraries

## Important Note

**Soundmentations currently supports mono audio only.** Any multichannel audio will be automatically converted to mono by taking the mean of all channels during loading. This ensures consistent processing across all transforms.

**What's New in v0.1.0:**
- 25+ audio transforms across time, amplitude, and frequency domains
- Gain transforms with envelope support (`Gain`, `RandomGain`, `PerSampleRandomGain`, `RandomGainEnvelope`)
- Advanced composition with `OneOf` for randomized transform selection
- Comprehensive documentation and examples
- Production-ready with extensive testing and validation

**Coming Soon:**
- Effect transforms (reverb, echo, distortion)
- Advanced noise injection and filtering
- Bounding box support for audio annotations
- Multichannel audio support
- Spectral transformations (mel-frequency, MFCC)

## Features

- **Time-based transforms**: Trim, pad, mask, and manipulate audio timing with precision
- **Amplitude transforms**: Control volume, gain, limiting, fading, and dynamic range
- **Frequency transforms**: Pitch shifting and frequency domain modifications
- **Probabilistic augmentation**: Apply transforms with configurable probability for robust datasets
- **Chainable pipelines**: Use `Compose` for sequential transforms or `OneOf` for random selection
- **NumPy compatible**: Works seamlessly with numpy arrays and scikit-learn pipelines
- **Mono audio focused**: Optimized for single-channel audio processing
- **High performance**: Optimized implementations for fast batch processing

## Installation

### Production Installation

```bash
pip install soundmentations
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/saumyarr8/soundmentations.git
cd soundmentations

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"  # Includes testing and documentation tools
```

### Requirements

- **Python**: 3.9+
- **NumPy**: Core array operations
- **SciPy**: Advanced signal processing
- **soundfile**: Audio file I/O (optional, for `load_audio` utility)

## Quick Start

```python
import numpy as np
from soundmentations import Compose, RandomTrim, Gain, FadeIn, OneOf

# Load your audio data (as numpy array)
audio_samples = np.random.randn(44100)  # 1 second of audio at 44.1kHz
sample_rate = 44100

# Create an augmentation pipeline
augment = Compose([
    RandomTrim(duration=(0.5, 2.0), p=0.8),     # Random trim with 80% probability
    OneOf([                                      # Randomly choose one transform
        Gain(gain=6.0),                          # +6dB boost
        Gain(gain=-6.0),                         # -6dB attenuation  
        FadeIn(duration=0.1),                    # Quick fade in
    ], p=0.7),
    Gain(gain=(-3, 3), p=0.5)                   # Final random gain adjustment
])

# Apply augmentations to audio
augmented_samples = augment(samples=audio_samples, sample_rate=sample_rate)
print(f"Original shape: {audio_samples.shape}")
print(f"Augmented shape: {augmented_samples.shape}")
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

### Time transforms

#### Trim transforms
- `Trim`
- `RandomTrim`
- `StartTrim`
- `EndTrim`
- `CenterTrim`

#### Pad transforms
- `Pad`
- `CenterPad`
- `StartPad`
- `PadToLength`
- `CenterPadToLength`
- `PadToMultiple`

### Amplitude transforms

#### Gain transforms
- `Gain`
- `RandomGain`

#### Limiter transforms
- `Limiter`

#### Fade transforms
- `FadeIn`
- `FadeOut`

### Frequency transforms

#### Pitch transforms
- `PitchShift`
- `RandomPitchShift`

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

### Core Transform Classes

#### Base Classes
```python
from soundmentations.core.base import BaseTransform, BaseTrim, BasePad, BaseGain

# All transforms inherit from BaseTransform and accept:
# - samples: numpy.ndarray (1D audio data)
# - sample_rate: int (audio sample rate)
# - p: float (probability of applying transform, default 1.0)
```

#### Time Domain Transforms
```python
from soundmentations import Trim, RandomTrim, Pad, CenterPad

# Precise trimming
trim = Trim(start_time=1.0, end_time=3.0)  # Trim to seconds 1-3

# Random duration trimming  
random_trim = RandomTrim(duration=(0.5, 2.0))  # Random duration between 0.5-2.0s

# Padding operations
pad = Pad(pad_length=44100, mode="zeros")  # Add 1 second of silence
center_pad = CenterPad(pad_length=22050)   # Add padding around center
```

#### Amplitude Transforms
```python
from soundmentations import Gain, RandomGain, PerSampleRandomGain

# Fixed gain adjustment
gain = Gain(gain=6.0)  # +6dB amplification

# Random gain range
random_gain = RandomGain(gain=(-6, 6))  # Random gain between -6dB and +6dB

# Per-sample random gain (advanced)
per_sample = PerSampleRandomGain(gain_range=(-0.1, 0.1))  # Subtle per-sample variation
```

#### Composition Classes
```python
from soundmentations import Compose, OneOf

# Sequential application
compose = Compose([
    RandomTrim(duration=(1.0, 2.0)),
    Gain(gain=3.0),
    Pad(pad_length=44100)
])

# Random selection
one_of = OneOf([
    Gain(gain=6.0),
    Gain(gain=-6.0),
    Trim(start_time=0.5, end_time=1.5)
], p=0.8)  # 80% chance to apply one of these transforms
```

### Utility Functions
```python
from soundmentations.utils.audio import load_audio

# Load audio file (automatically converts to mono)
samples, sample_rate = load_audio("audio.wav")
print(samples.shape)  # (n_samples,) - always 1D

# Validate audio input
from soundmentations.utils.validation import validate_audio
validate_audio(samples)  # Raises exception if invalid format
```

## Requirements

- **Python**: 3.9+
- **NumPy**: Core array operations
- **SciPy**: Advanced signal processing  
- **soundfile**: Audio file I/O (optional, for `load_audio` utility)

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest tests/ --cov=soundmentations --cov-report=html

# Run specific test file
python -m pytest tests/test_gain.py -v
```

## Documentation

Visit our [documentation site](https://saumyarr8.github.io/soundmentations/) for:
- Complete API reference
- Tutorial notebooks
- Advanced usage examples  
- Developer guides

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

**Soundmentations** - Making audio augmentation simple and powerful!

**Star this repository** to stay updated on new features and releases!

**Need Help?** Open an issue or check our [documentation](https://saumyarr8.github.io/soundmentations/)