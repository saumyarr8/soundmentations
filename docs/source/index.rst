.. soundmentations documentation master file, created by
   sphinx-quickstart on Wed Jun 18 16:29:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Soundmentations Documentation!
==========================================

**Soundmentations** is a Python library for audio data augmentation and sound classification. 
It provides a collection of audio transforms that can be used to augment audio datasets 
for machine learning applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Features
--------

* **Audio Transforms**: Pitch shifting, fading, limiting, and trimming
* **Composition**: Chain multiple transforms together  
* **Probability Control**: Apply transforms with specified probabilities
* **Easy Integration**: Works seamlessly with NumPy arrays
* **Extensible**: Simple base classes for creating custom transforms

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from soundmentations import PitchShift, FadeIn, Compose

   # Create a composition of transforms
   transform = Compose([
       PitchShift(semitones=2.0, p=0.8),
       FadeIn(duration=0.1, p=0.5),
   ])

   # Apply to audio
   audio = np.random.randn(44100)  # 1 second of audio
   augmented = transform(audio, sample_rate=44100)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

