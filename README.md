# crosscorr
A suite of functions that can compute various cosmological correlation functions, written in C and packaged into a python module

**crosscorr** is a python module designed for use in the Dark Energy Spectroscopic Instrument (DESI) survey, specifically for data from the Peculiar Velocity (PV) survey.

**crosscorr** is built upon a foundation of C code, used to compute the auto-correlation functions and cross-correlation function of galaxy clustering and peculiar velocity statistics.
The module contains functions that are applicable to simulation data -- where the full 3D velocity field is known -- and to observational data -- where only the radial component of the velocity is measurable, and is capable of handling cases where the effects of redshift-space distortions (RSDs) are either included or ignored.

## Installation
1. Clone this repo by pasting the following into your terminal

        git clone https://github.com/r-jturner/crosscorr.git

2. Inside /src/, run the Makefile using 'make' - you may need to modify the Makefile to match your compiler (this compiler must be able to handle OpenMP code). This will compile the C code in the repo and create a shared object (.so file) that is required by the python wrapper.

3. It should then be possible to make python scripts within the directory using the crosscorr functions, as long as you import 'corr_desi_wrapper' along with your usual packages.

## Usage
I will add detailed examples, but see the notebook 'crosscorr_example.ipynb' for a rudimentary look into the functionality of **crosscorr**

## Support
For support, contact Ryan Turner
