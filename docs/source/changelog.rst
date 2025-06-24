Changelog
=========

Version 0.1.0 (2025-01-24)
---------------------------

**Initial Release**

New Features:
~~~~~~~~~~~~~

* **Pitch Transforms**
  - ``PitchShift``: Fixed pitch shifting
  - ``RandomPitchShift``: Random pitch variations

* **Amplitude Transforms**
  - ``Limiter``: Hard audio limiting
  - ``FadeIn``: Fade-in effects
  - ``FadeOut``: Fade-out effects

* **Trim Transforms**
  - ``Trim``: Basic audio trimming
  - ``RandomTrim``: Random duration trimming
  - ``StartTrim``, ``EndTrim``, ``CenterTrim``: Directional trimming

* **Core Features**
  - ``Compose``: Chain multiple transforms
  - Probability control for all transforms
  - Comprehensive error handling
  - NumPy array support

Documentation:
~~~~~~~~~~~~~~

* Complete API reference
* Installation guide
* Tutorials and examples
* Professional Sphinx documentation with Furo theme