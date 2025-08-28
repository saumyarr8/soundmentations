"""Audio transformations organized by category."""

# Import from time-based transforms
from .time import (
    # Trim transforms
    Trim,
    RandomTrim,
    StartTrim,
    EndTrim,
    CenterTrim,
    
    # Pad transforms
    Pad,
    CenterPad,
    StartPad,
    PadToLength,
    CenterPadToLength,
    PadToMultiple,

    # Mask transforms
    Mask,
)

# Import from amplitude-based transforms
from .amplitude import (
    # Gain transforms
    Gain,
    RandomGain,
    PerSampleRandomGain,
    RandomGainEnvelope,

    # Limiter transforms
    Limiter,

    # Fade transforms
    FadeIn,
    FadeOut,

    # Compressor transforms
    Compressor,
)

# Import from frequency-based transforms
from .frequency import (
    # Pitch transforms
    PitchShift,
    RandomPitchShift,
)

# Export all transforms for public API
__all__ = [
    # Time transforms
    # Trim transforms
    "Trim",
    "RandomTrim", 
    "StartTrim",
    "EndTrim",
    "CenterTrim",
    
    # Pad transforms
    "Pad",
    "CenterPad",
    "StartPad",
    "PadToLength",
    "CenterPadToLength",
    "PadToMultiple",

    #Mask transforms
    "Mask",



    # Amplitude transforms
    # Gain transforms
    "Gain",
    "RandomGain",
    "PerSampleRandomGain",
    "RandomGainEnvelope",

    # Limiter transforms
    "Limiter",

    # Fade transforms
    "FadeIn",
    "FadeOut",

    # Compressor transforms
    "Compressor",



    # Frequency transforms
    # Pitch-shift transforms
    "PitchShift",
    "RandomPitchShift",
]