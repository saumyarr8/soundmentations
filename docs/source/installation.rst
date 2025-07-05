Installation Guide
==================

System Requirements
-------------------

**Operating Systems:**
- Windows 10/11 (64-bit)
- macOS 10.15+ (Intel/Apple Silicon)
- Linux (Ubuntu 18.04+, CentOS 7+)

**Python Requirements:**
- Python 3.9 or higher
- 64-bit Python installation recommended

Core Dependencies
-----------------

Soundmentations requires the following packages:

.. list-table::
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - numpy
     - ≥2.0.0
     - Array operations
   * - scipy
     - ≥1.0.0
     - Signal processing
   * - librosa
     - ≥0.10.0
     - Audio analysis
   * - soundfile
     - ≥0.10.0
     - Audio I/O

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/saumyarr8/soundmentations.git
   cd soundmentations
   
   # Install in development mode
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributing or development:

.. code-block:: bash

   # Install with development dependencies
   pip install -e ".[dev]"
   
   # Or install specific extras
   pip install -e ".[docs]"    # Documentation tools
   pip install -e ".[test]"    # Testing tools

Using Conda
~~~~~~~~~~~

If you prefer conda for dependency management:

.. code-block:: bash

   # Create conda environment
   conda create -n soundmentations python=3.10
   conda activate soundmentations
   
   # Install core dependencies
   conda install -c conda-forge numpy scipy librosa soundfile
   
   # Install soundmentations
   pip install -e .

Verify Installation
-------------------

Test your installation:

.. code-block:: python

   import soundmentations
   print(f"Soundmentations version: {soundmentations.__version__}")
   
   # Test basic functionality
   from soundmentations import PitchShift
   import numpy as np
   
   # Create test audio
   audio = np.random.randn(44100)
   
   # Apply transform
   transform = PitchShift(semitones=2.0)
   result = transform(audio, sample_rate=44100)
   
   print("✅ Installation successful!")
   print(f"Input shape: {audio.shape}")
   print(f"Output shape: {result.shape}")

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. librosa installation fails:**

.. code-block:: bash

   # Try installing with conda first
   conda install -c conda-forge librosa
   
   # Or install system dependencies (Ubuntu/Debian)
   sudo apt-get install libsndfile1-dev ffmpeg
   
   # Then retry pip install
   pip install librosa

**2. soundfile issues on Windows:**

.. code-block:: bash

   # Install Microsoft Visual C++ Redistributable
   # Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   
   # Or use conda
   conda install -c conda-forge soundfile

**3. NumPy version conflicts:**

.. code-block:: bash

   # Upgrade to latest NumPy
   pip install --upgrade numpy>=2.0.0

**4. Import errors:**

.. code-block:: python

   # Check Python path
   import sys
   print(sys.path)
   
   # Check if package is installed
   import pkg_resources
   pkg_resources.get_distribution('soundmentations')

**5. Permission errors (Linux/macOS):**

.. code-block:: bash

   # Use --user flag
   pip install --user -e .
   
   # Or create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows

Optional Dependencies
---------------------

For additional functionality:

.. code-block:: bash

   # For advanced audio analysis
   pip install librosa[complete]
   
   # For visualization
   pip install matplotlib seaborn
   
   # For Jupyter notebooks
   pip install jupyter ipython
   
   # For audio format support
   pip install pydub

Development Setup
-----------------

Complete development environment:

.. code-block:: bash

   # Clone and setup
   git clone https://github.com/saumyarr8/soundmentations.git
   cd soundmentations
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   
   # Install with all dependencies
   pip install -e ".[dev,docs,test]"
   
   # Verify development setup
   pytest tests/
   sphinx-build docs/source docs/build