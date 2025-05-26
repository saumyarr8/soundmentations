# Soundmentations

A Python library for audio data augmentation and transformations.

## Quick Start

```python
from soundmentations import Compose, Trim

# Create an augmentation pipeline
augment = Compose([
    Trim(start_time=1, end_time=3)
])

# Apply augmentations to audio
augmented_samples = augment(samples=audio_samples, sample_rate=sample_rate)
```

## Features

- Easy-to-use audio augmentation techniques
- Compatible with numpy arrays
- Chainable transformations

## Documentation

See the [documentation](https://saumyarr8.github.io/soundmentations/) for detailed usage examples and API reference.

## License

MIT License