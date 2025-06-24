Examples
========

Complete Examples
-----------------

Data Augmentation for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from soundmentations import Compose, PitchShift, FadeIn, Limiter
   from soundmentations.utils import load_audio

   # Load audio file
   audio, sample_rate = load_audio("speech.wav")
   
   # Create augmentation pipeline
   augment = Compose([
       PitchShift(semitones=(-2, 2), p=0.8),    # Random pitch variation
       FadeIn(duration=0.1, p=0.3),             # Sometimes fade in
       Limiter(threshold=0.9, p=1.0),           # Always apply limiting
   ])

   # Generate 10 variations
   variations = []
   for i in range(10):
       augmented = augment(audio, sample_rate)
       variations.append(augmented)

Audio Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import librosa
   from soundmentations import *

   # Load and process multiple files
   for audio_file in ["file1.wav", "file2.wav", "file3.wav"]:
       audio, sr = librosa.load(audio_file, sr=44100)
       
       # Apply different effects
       pitch_shifted = PitchShift(semitones=5.0)(audio, sr)
       faded = FadeIn(duration=0.5)(audio, sr)
       limited = Limiter(threshold=0.8)(audio, sr)
       
       # Save results
       librosa.output.write_wav(f"processed_{audio_file}", pitch_shifted, sr)

Real-world Use Cases
~~~~~~~~~~~~~~~~~~~~

**Music Data Augmentation**

.. code-block:: python

   # For music classification
   music_augment = Compose([
       RandomPitchShift(min_semitones=-3, max_semitones=3, p=0.7),
       FadeIn(duration=0.2, p=0.2),
       FadeOut(duration=0.2, p=0.2),
   ])

**Speech Enhancement**

.. code-block:: python

   # For speech processing
   speech_enhance = Compose([
       Limiter(threshold=0.95, p=1.0),
       FadeIn(duration=0.05, p=1.0),
       FadeOut(duration=0.05, p=1.0),
   ])