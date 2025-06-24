"""
Pitch manipulation transforms for audio data augmentation.

This module provides transforms for modifying the pitch of audio signals
without changing the duration. These are useful for data augmentation
in audio processing tasks.

Available transforms:
- PitchShift: Shift pitch by a fixed number of semitones
- RandomPitchShift: Randomly shift pitch within a specified range
"""

from .pitch_shift import (
    PitchShift,
    RandomPitchShift,
    )


__all__ = [
    # Pitch transforms
    "PitchShift",
    "RandomPitchShift", 
]