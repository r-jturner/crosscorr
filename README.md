# crosscorr
A suite of functions that can compute various cosmological correlation functions, written in C and packaged into a python module

**crosscorr** is a python module designed for use in the Dark Energy Spectroscopic Instrument (DESI) survey, specifically for data from the Peculiar Velocity (PV) survey.

**crosscorr** is built upon a foundation of C code, used to compute the auto-correlation functions and cross-correlation function of galaxy clustering and peculiar velocity statistics.
The module contains functions that are applicable to simulation data -- where the full 3D velocity field is known -- and to observational data -- where only the radial component of the velocity is measurable, and is capable of handling cases where the effects of redshift-space distortions (RSDs) are either included or ignored.
