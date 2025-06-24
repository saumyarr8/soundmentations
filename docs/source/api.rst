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

   ~soundmentations.PitchShift
   ~soundmentations.RandomPitchShift
   ~soundmentations.Limiter
   ~soundmentations.FadeIn
   ~soundmentations.FadeOut
   ~soundmentations.Trim
   ~soundmentations.RandomTrim
   ~soundmentations.StartTrim
   ~soundmentations.EndTrim
   ~soundmentations.CenterTrim
   ~soundmentations.Compose

Transforms by Category
----------------------

Pitch Transforms
~~~~~~~~~~~~~~~~

.. currentmodule:: soundmentations

.. autosummary::
   :toctree: _autosummary

   PitchShift
   RandomPitchShift

Amplitude Transforms
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   Limiter
   FadeIn
   FadeOut

Trim Transforms
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   Trim
   RandomTrim
   StartTrim
   EndTrim
   CenterTrim

Composition
-----------

.. autosummary::
   :toctree: _autosummary

   Compose