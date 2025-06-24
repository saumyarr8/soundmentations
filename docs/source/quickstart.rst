Quick Start Guide
=================

Basic Usage
-----------

Import and use a single transform:

.. code-block:: python

   import numpy as np
   from soundmentations import PitchShift

   # Load your audio (as numpy array)
   audio = np.random.randn(44100)  # 1 second of random audio
   sample_rate = 44100

   # Create transform
   pitch_shift = PitchShift(semitones=2.0, p=1.0)

   # Apply transform
   augmented_audio = pitch_shift(audio, sample_rate)

Using Multiple Transforms
-------------------------

Chain multiple transforms using Compose:

.. code-block:: python

   from soundmentations import Compose, PitchShift, FadeIn, Limiter

   # Create a composition
   transform = Compose([
       PitchShift(semitones=2.0, p=0.8),
       FadeIn(duration=0.1, p=0.5),
       Limiter(threshold=0.9, p=1.0),
   ])

   # Apply to audio
   augmented = transform(audio, sample_rate)

Available Transforms
--------------------

**Pitch Transforms**

* :class:`~soundmentations.PitchShift` - Shift pitch by fixed semitones
* :class:`~soundmentations.RandomPitchShift` - Random pitch shifting

**Amplitude Transforms**

* :class:`~soundmentations.Limiter` - Hard limiting
* :class:`~soundmentations.FadeIn` - Fade in effect
* :class:`~soundmentations.FadeOut` - Fade out effect

**Trim Transforms**

* :class:`~soundmentations.Trim` - Trim audio to specific duration
* :class:`~soundmentations.RandomTrim` - Random trimming