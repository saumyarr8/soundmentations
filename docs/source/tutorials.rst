Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorials/getting_started
   tutorials/advanced_usage
   tutorials/custom_transforms

Getting Started Tutorial
========================

This tutorial walks you through the basics of using Soundmentations.

Step 1: Basic Transform
-----------------------

.. code-block:: python

   import numpy as np
   from soundmentations import PitchShift

   # Create sample audio (1 second of sine wave)
   sample_rate = 44100
   duration = 1.0
   t = np.linspace(0, duration, int(sample_rate * duration))
   audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)

   # Apply pitch shift (shift up by 7 semitones - perfect fifth)
   pitch_shift = PitchShift(semitones=7.0, p=1.0)
   shifted_audio = pitch_shift(audio, sample_rate)

Step 2: Multiple Transforms
----------------------------

.. code-block:: python

   from soundmentations import Compose, FadeIn, Limiter

   # Chain multiple transforms
   transform_chain = Compose([
       PitchShift(semitones=4.0, p=1.0),     # Major third up
       FadeIn(duration=0.1, p=1.0),          # Fade in over 100ms
       Limiter(threshold=0.8, p=1.0),        # Limit to prevent clipping
   ])

   # Apply the entire chain
   processed_audio = transform_chain(audio, sample_rate)

Step 3: Random Augmentation
----------------------------

.. code-block:: python

   from soundmentations import RandomPitchShift

   # Create random variations for data augmentation
   random_augment = Compose([
       RandomPitchShift(min_semitones=-2, max_semitones=2, p=0.8),
       FadeIn(duration=0.05, p=0.3),
       Limiter(threshold=0.9, p=1.0),
   ])

   # Generate 5 different variations
   variations = []
   for i in range(5):
       variation = random_augment(audio, sample_rate)
       variations.append(variation)
       print(f"Generated variation {i+1}")