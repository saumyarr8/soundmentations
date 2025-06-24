Installation Guide
==================

Requirements
------------

**System Requirements:**
- Python 3.9 or higher
- 64-bit operating system (Windows, macOS, or Linux)

**Core Dependencies:**
- NumPy >= 2.0.0
- SciPy >= 1.0.0  
- librosa >= 0.10.0
- soundfile >= 0.10.0

Quick Installation
------------------

Install from PyPI (when available):

.. code-block:: bash

   pip install soundmentations

Development Installation
------------------------

For the latest features and development:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/saumyarr8/soundmentations.git
   cd soundmentations
   
   # Install in development mode
   pip install -e .
   
   # Or with development dependencies
   pip install -e ".[dev]"

Verify Installation
-------------------

.. code-block:: python

   import soundmentations
   print(soundmentations.__version__)
   
   # Test basic functionality
   from soundmentations import PitchShift
   import numpy as np
   
   audio = np.random.randn(44100)
   transform = PitchShift(semitones=2.0)
   result = transform(audio, sample_rate=44100)
   print("Installation successful!")

Optional Dependencies
---------------------

For additional functionality:

.. code-block:: bash

   # For advanced audio I/O
   pip install librosa[complete]
   
   # For visualization (if you plan to plot audio)
   pip install matplotlib
   
   # For Jupyter notebook examples
   pip install jupyter ipython

Troubleshooting
---------------

**Common Issues:**

1. **librosa installation fails**: Try installing with conda:
   
   .. code-block:: bash
   
      conda install -c conda-forge librosa

2. **soundfile issues on Windows**: Install Microsoft Visual C++ Redistributable

3. **Import errors**: Make sure you're using the correct Python environment