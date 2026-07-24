# crosscorr
A suite of functions for computing cosmological galaxy and velocity correlation functions, written in C with a python interface

**crosscorr** was developed for use with the Dark Energy Spectroscopic Instrument (DESI) Peculiar Velocity (PV) survey. The bulk of the computational work is implemented in C with OpenMP parallelisation for more efficient galaxy pair counting. Python is used to provide user-friendly functions for scientific analysis.

The package supports:
- Galaxy auto-correlations
- Velocity auto-correlations
- Galaxy-velocity cross-correlations
- Radial velocity analysis (observational data)
- 3D velocity analysis (simulation data)
- Correlations in (r) space and (r, mu) space
- Multipole decompositions of the correlation functions

## Repo structure
```text
crosscorr/
├── setup.py                 # C extension configuration
├── pyproject.toml           # Project metadata and build configuration
├── README.md
├── LICENSE
├── MANIFEST.in
├── include/
│   └── corr_desi.h
├── csrc/                    # C source code
│   ├── corr_desi.c
│   ├── corr_desi_smu.c
│   ├── corr_desi_xyz.c
│   ├── Makefile
│   └── obj/                 # Compiled objects (build.sh output)
├── src/
│   └── crosscorr/           # Python package
│       ├── __init__.py
│       ├── corr_desi_wrapper.py
│       └── obj/             # Compiled .so files (pip install output)
├── examples/
│   └── crosscorr_example.ipynb
└── tests/
```
The C source code is located in `csrc/`, while the Python package is located in `src/crosscorr/`. The compiled extension can be found in `src/crosscorr/obj/` after installation.


## Installation

### Quick Start
The C extension is automatically compiled during installation. Just run:

#### Linux
```bash
git clone https://github.com/r-jturner/crosscorr.git
cd crosscorr
pip install -e .
```

#### macOS
Apple clang doesn't provide OpenMP support by default. Install GCC first:
```bash
brew install gcc
git clone https://github.com/r-jturner/crosscorr.git
cd crosscorr
CC=gcc-16 pip install -e .
```
(Replace `gcc-16` with whichever GCC version is in `/opt/homebrew/bin/`)

### Verify Installation
```python
import crosscorr
print(crosscorr.__version__)
```

### Development Installation (Optional)
If you're modifying the C code and want faster iteration, you can compile directly:
```bash
cd csrc
make CC=gcc-16 rebuild  # macOS; omit CC= on Linux
cd ..
```

Then modify the C code and re-import. Or reinstall with `CC=gcc-16 pip install -e .` (macOS).

---

## Usage
See the notebook 
```text
examples/crosscorr_example.ipynb
```
for a rudimentary look into the functionality of **crosscorr**. More detailed examples will be added in future.

Briefly, *corrPairCount* is used to measure correlations in (s) space, while *corrPairCount_smu* is used to measure correlations in (s, $\mu$) space. By construction, *corrPairCount_smu* can only accept radial velocity data, while *corrPairCount* can accept radial velocities or 3D velocity information.

There are several pair counting estimators included, including the Peebles & Davis estimator and the Landy & Szalay estimator. Users should feel free to use the numerators and denominators produced by the pair-counting functions to create their own estimators for their own purposes, too.

The *calc_XXXX* functions can be used to measure any of the galaxy or velocity auto- or cross-correlations, using the outputs from the *corrPairCount* functions. The *multipole_XXXX* functions can take the final (s,$\mu$) correlation function measurements produced by these functions and decompose them into their non-zero multipole moments.

## Support
For support, contact: 

Ryan Turner 
ryan.turner@anu.edu.au
