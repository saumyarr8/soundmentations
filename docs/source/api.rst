API Reference
=============

.. currentmodule:: soundmentations

Quick Import Guide
------------------

Import all transforms directly from the main package:

.. code-block:: python

   from soundmentations import *

All Available Transforms
-------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: class.rst

   Compose
   OneOf
   Trim
   RandomTrim
   StartTrim
   EndTrim
   CenterTrim
   Pad
   CenterPad
   StartPad
   PadToLength
   CenterPadToLength
   PadToMultiple
   Mask
   Gain
   RandomGain
   PerSampleRandomGain
   RandomGainEnvelope
   Limiter
   FadeIn
   FadeOut
   Compressor
   PitchShift
   RandomPitchShift

Transforms by Category
----------------------

Composition
~~~~~~~~~~~

.. currentmodule:: soundmentations

.. autosummary::
   :toctree: _autosummary

   Compose
   OneOf

Time Transforms
~~~~~~~~~~~~~~~

Trim Transforms
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   Trim
   RandomTrim
   StartTrim
   EndTrim
   CenterTrim

Pad Transforms
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   Pad
   CenterPad
   StartPad
   PadToLength
   CenterPadToLength
   PadToMultiple

Mask Transforms
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   Mask

Amplitude Transforms
~~~~~~~~~~~~~~~~~~~~

Gain Transforms
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   Gain
   RandomGain
   PerSampleRandomGain
   RandomGainEnvelope

Limiter Transforms
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   Limiter

Fade Transforms
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   FadeIn
   FadeOut

Compressor Transforms
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   Compressor

Frequency Transforms
~~~~~~~~~~~~~~~~~~~~

Pitch Transforms
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   PitchShift
   RandomPitchShift