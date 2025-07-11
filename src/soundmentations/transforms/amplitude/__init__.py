"""
Amplitude transformation module for audio processing.

This module provides various transforms that operate on the amplitude
envelope of audio signals, allowing for dynamic range adjustment,
normalization, and other amplitude-based effects.
"""
from .gain import (
    Gain,
)

from .limiter import (
    Limiter,
)

from .fade import (
    FadeIn,
    FadeOut,
)

# Export all transforms for public API
__all__ = [
    "Gain",
    "Limiter",
    "FadeIn",
    "FadeOut",
]