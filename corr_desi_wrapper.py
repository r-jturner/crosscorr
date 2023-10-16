# Import necessary modules
import ctypes as cts
import numpy as np
from numpy.ctypeslib import ndpointer

_corr = cts.CDLL('src/obj/corrdesi.so')

# Define a corr_smu function, and then the equivalent (x,y,z) functions too

def corrPairCount(sample1, sample2, smax, swidth, estimator, weights1 = None, weights2 = None, vel = "3D", nthreads = 1):
    """
    Python function wrapping C function 'pairCounter' that computes estimators
    of the 2PCF, v-v auto-corr and g-v cross-corr functions.

    Parameters
    -----------
    sample1 : array with shape (N,4),
            (x,y,z,u) - positions in cartesian coordinates and radial velocity
    sample2 : see above
    smax : maximum separation to consider when computing correlation estimates (Mpc/h)
    swidth : width of separation bins (Mpc/h)
    estimator : string that determines which estimator to compute,
                "psi1", "psi2", "psi3" or "xiGG"
    nthreads : number of threads to use when executing function in parallel,
               set by default to 1

    Outputs
    -----------
    numerator : 1D array with length N = (int)(smax / swidth),
                each bin contains the summed total of the numerator of the specified estimator for every
                pair of galaxies that fall in that separation bin
    denominator : as above, but instead contains the summed total of the denominator for the estimator

    In the case 'estimator = "xiGG"' the denominator is always 0 as this is essentially a pair counting algorithm 
    with no pairwise weighting needed, and so the numerator captures all required information.
    For more information on the numerator and denominator of these estimators, see Turner et al. (2021)
    or Turner et al. (2023).
    """
    
    # Check that the catalogs provided have the correct shape, throw error if not
    # Also store the total number of objects in each catalog to pass to C function
    radial_check = 0
    if (vel == "3D"): 
        ncol = 6
        radial_check = 0
    elif (vel == "u"): 
        ncol = 4
        radial_check = 1
    else: raise Exception("'vel' argument must take either '3D' or 'u' as value, to indicate if the velocity information is three-dimensional or radial.")
    if (sample1.shape[1] != ncol or sample2.shape[1] != ncol):
        raise Exception("Please check that the number of columns is correct - if xyz == True, you must provide (vx, vy, vz) velocity info. as well as (x,y,z) positions.")
    len_sample1 = sample1.shape[0]
    len_sample2 = sample2.shape[0]

    # If samples are not equivalent, set parameter 'equiv' to 0
    equiv = 0
    if (len_sample1 != len_sample2):
        equiv = 0
    elif ((sample1 == sample2).all()):
        # Otherwise set 'equiv' to 1
        equiv = 1

    # Check if weight vectors have been supplied or not, if not set weights to be a vector of ones
    if weights1 == None:
        weights1 = np.ones(len_sample1)
    if weights2 == None:
        weights2 = np.ones(len_sample2)
    
    # Prepare inputs to be passed to C
    sample1, sample2 = np.ascontiguousarray(sample1), np.ascontiguousarray(sample2)
    weights1, weights2 = np.ascontiguousarray(weights1), np.ascontiguousarray(weights2)
    c_estimator = estimator.encode('utf-8')

    # Define a class to hold the outputs of the 'lin_corr' struct
    # This needs to be a global definition, otherwise we cannot access the contents
    # of the class outside of this function (which is not ideal)
    #
    # We also need to dynamically define how large each vector in the class should be,
    # as this can change depending on the width of our bins, and the maximum separation
    # we want to consider - use numBins to determine the length of these vectors
    numBins = (np.ceil(smax/swidth)).astype(int)
    global pairCounts
    class pairCounts(cts.Structure):
        _fields_ = [('num',cts.POINTER(cts.c_double * (numBins))),
                    ('den',cts.POINTER(cts.c_double * (numBins)))]
        
    # Define the ctypes inputs and outputs for the C function 'pairCounter'
    _corr.pairCounter.argtypes = [cts.c_int, 
                                  cts.c_int, 
                                  cts.c_int,
                                  ndpointer(dtype=np.float64, ndim=2, shape=(len_sample1, ncol), flags='C_CONTIGUOUS'),
                                  ndpointer(dtype=np.float64, ndim=2, shape=(len_sample2, ncol), flags='C_CONTIGUOUS'),
                                  ndpointer(dtype=np.float64, ndim=1, shape=len_sample1, flags="C_CONTIGUOUS"),
                                  ndpointer(dtype=np.float64, ndim=1, shape=len_sample2, flags="C_CONTIGUOUS"),
                                  cts.c_int,
                                  cts.c_int,
                                  cts.c_char_p,
                                  cts.c_int]
    _corr.pairCounter_xyz.argtypes = [cts.c_int, 
                                  cts.c_int, 
                                  cts.c_int,
                                  ndpointer(dtype=np.float64, ndim=2, shape=(len_sample1, ncol), flags='C_CONTIGUOUS'),
                                  ndpointer(dtype=np.float64, ndim=2, shape=(len_sample2, ncol), flags='C_CONTIGUOUS'),
                                  ndpointer(dtype=np.float64, ndim=1, shape=len_sample1, flags="C_CONTIGUOUS"),
                                  ndpointer(dtype=np.float64, ndim=1, shape=len_sample2, flags="C_CONTIGUOUS"),
                                  cts.c_int,
                                  cts.c_int,
                                  cts.c_char_p,
                                  cts.c_int]
    # We expect the function 'pairCounter' to return a pointer to a 'pairCounts' type of python class
    _corr.pairCounter.restype = cts.POINTER(pairCounts)
    _corr.pairCounter_xyz.restype = cts.POINTER(pairCounts)

    # Run the function 'pairCounter'
    if (radial_check == 0):
        result = _corr.pairCounter_xyz(len_sample1, len_sample2, equiv, sample1, sample2,
                                    weights1, weights2, smax, swidth,  c_estimator, nthreads)
    elif (radial_check == 1):
        result = _corr.pairCounter(len_sample1, len_sample2, equiv, sample1, sample2,
                                    weights1, weights2, smax, swidth, c_estimator, nthreads)
    
    # Separately save the elements of the result to numpy arrays
    numerator = np.array(result[0].num[0][:])
    denominator = np.array(result[0].den[0][:])

    # Since we dynamically allocated memory to the structure (and its arrays) based on the number
    # of objects in our samples, we have to free that memory ourselves to avoid leaks
    _corr.free_arraymemory(result[0].num)
    _corr.free_arraymemory(result[0].den)
    _corr.free_structmemory(result)

    # Finally, return our desired outputs
    return numerator, denominator


##
## can we remove psi1 and psi2 from the "_smu" functions? 
##

def corrPairCount_smu(sample1, sample2, smax, swidth, muwidth, estimator, weights1 = None, weights2 = None, nthreads=1):
    """
    Python function wrapping C function 'pairCounter' that computes estimators
    of the 2PCF, v-v auto-corr and g-v cross-corr functions.

    Parameters
    -----------
    sample1     : array with shape (N,4),
                (x,y,z,u) - positions in cartesian coordinates and radial velocity
    sample2     : see above
    smax        : maximum separation to consider when computing correlation estimates (Mpc/h)
    swidth      : width of separation bins (Mpc/h)
    muwidth     : width of cos(theta_mu) bins
    estimator   : string that determines which estimator to compute,
                "psi1", "psi2", "psi3" or "xiGG"
    nthreads    : number of threads to use when executing function in parallel,
                set by default to 1

    Outputs
    -----------
    numerator : 1D array with length N = (int)(smax / swidth),
                each bin contains the summed total of the numerator of the specified estimator for every
                pair of galaxies that fall in that separation bin
    denominator : as above, but instead contains the summed total of the denominator for the estimator

    In the case 'estimator = "xiGG"' the denominator is always 0 as this is essentially a pair counting algorithm 
    with no pairwise weighting needed, and so the numerator captures all required information.
    For more information on the numerator and denominator of these estimators, see Turner et al. (2021)
    or Turner et al. (2023).
    """
    
    # Check that the catalogs provided have the correct shape, throw error if not
    # Also store the total number of objects in each catalog to pass to C function
    if (sample1.shape[1] != 4 or sample2.shape[1] != 4):
        raise Exception("Both datasets should have 4 columns - x,y,z positions and a radial velocity.")
    len_sample1 = sample1.shape[0]
    len_sample2 = sample2.shape[0]

    # If samples are not equivalent, set parameter 'equiv' to 0
    equiv = 0
    if (len_sample1 != len_sample2):
        equiv = 0
    elif ((sample1 == sample2).all()):
        # Otherwise set 'equiv' to 1
        equiv = 1

    # Check if weight vectors have been supplied or not, if not set weights to be a vector of ones
    if weights1 == None:
        weights1 = np.ones(len_sample1)
    if weights2 == None:
        weights2 = np.ones(len_sample2)
    
    # Prepare inputs to be passed to C
    sample1, sample2 = np.ascontiguousarray(sample1), np.ascontiguousarray(sample2)
    weights1, weights2 = np.ascontiguousarray(weights1), np.ascontiguousarray(weights2)
    c_estimator = estimator.encode('utf-8')

    # Define a class to hold the outputs of the 'lin_corr' struct
    # This needs to be a global definition, otherwise we cannot access the contents
    # of the class outside of this function (which is not ideal)
    #
    # We also need to dynamically define how large each vector in the class should be,
    # as this can change depending on the width of our bins, and the maximum separation
    # we want to consider - use numBins to determine the length of these vectors
    numBins = (np.ceil(smax/swidth)).astype(int)
    muBins = int(2.0/muwidth)
    print("s bins = ", numBins, "and mu bins = ", muBins)
    global pairCounts
    class pairCounts(cts.Structure):
        _fields_ = [('num',cts.POINTER(cts.c_double * (numBins * muBins))),
                    ('den',cts.POINTER(cts.c_double * (numBins * muBins)))]
        
    # Define the ctypes inputs and outputs for the C function 'pairCounter'
    _corr.pairCounter_smu.argtypes = [cts.c_int, 
                                  cts.c_int, 
                                  cts.c_int,
                                  ndpointer(dtype=np.float64, ndim=2, shape=(len_sample1, 4), flags='C_CONTIGUOUS'),
                                  ndpointer(dtype=np.float64, ndim=2, shape=(len_sample2, 4), flags='C_CONTIGUOUS'),
                                  ndpointer(dtype=np.float64, ndim=1, shape=len_sample1, flags="C_CONTIGUOUS"),
                                  ndpointer(dtype=np.float64, ndim=1, shape=len_sample2, flags="C_CONTIGUOUS"),
                                  cts.c_int,
                                  cts.c_int,
                                  cts.c_double,
                                  cts.c_char_p,
                                  cts.c_int]
    # We expect the function 'pairCounter' to return a pointer to a 'pairCounts' type of python class
    _corr.pairCounter_smu.restype = cts.POINTER(pairCounts)

    # Run the function 'pairCounter'
    result = _corr.pairCounter_smu(len_sample1, len_sample2, equiv, sample1, sample2,
                               weights1, weights2, smax, swidth, muwidth, c_estimator, nthreads)
    
    # Separately save the elements of the result to numpy arrays
    numerator = np.array(result[0].num[0][:])
    denominator = np.array(result[0].den[0][:])

    # Since we dynamically allocated memory to the structure (and its arrays) based on the number
    # of objects in our samples, we have to free that memory ourselves to avoid leaks
    _corr.free_arraymemory(result[0].num)
    _corr.free_arraymemory(result[0].den)
    _corr.free_structmemory(result)

    # Finally, return our desired outputs
    return numerator, denominator

# Pair-counting functions (Peebles and Davis; Landy & Szalay; Turner, Blake, and Ruggeri)
def peebles(nD,nR,DD,RR):
    norm_sq = (nR**2) / (nD**2)
    output = np.where(RR != 0, norm_sq*(DD/RR) - 1, 0.0)
    return output #norm_sq*(DD/RR) - 1

def landy_szalay(nD,nR,DD,DR,RR):
    norm_sq = (nR**2) / (nD**2)
    norm = nR / nD
    output = np.where(RR != 0, norm_sq*(DD/RR) - norm*(DR/RR) + 1, 0.0)
    return output #norm_sq*(DD/RR) - norm*(DR/RR) + 1

def turner(nD,nR,DD,DR,RD,RRn,RRd):
    norm_sq = (nR**2) / (nD**2)
    norm = nR / nD
    output = np.where(RRd != 0, norm_sq*(DD/RRd) - norm*(DR/RRd) - norm*(RD/RRd) + RRn/RRd, 0.0)
    return output #norm_sq*(DD/RRd) - norm*(DR/RRd) - norm*(RD/RRd) + RRn/RRd

def vel_short(nD,nR,DD,RR):
    norm_sq = (nR**2) / (nD**2)
    output = np.where(RR != 0, norm_sq*(DD/RR), 0.0)
    return output #norm_sq*(DD/RR)

def vel_long(nD,nR,DD,DR,RRn,RRd):
    norm_sq = (nR**2) / (nD**2)
    norm = nR / nD
    return norm_sq*(DD/RRd) - norm*(DR/RRd) - RRn/RRd

# Estimators of the correlation functions
def calc_psi12(nD,nR,DD,DR,RRn,RRd,estimator = "long"):
    if (estimator == "long"):
        psi12 = vel_long(nD,nR,DD,DR,RRn,RRd)
        return psi12
    elif (estimator  == "short"):
        psi12 = vel_short(nD,nR,DD,RRd)
        return psi12
    else:
        return "psi1 and psi2 estimators must be either 'long' or 'short'"

def calc_psi3(nD,nR,DD,DR,RD,RRn,RRd,estimator = "turner"):
    if (estimator == "short"):
        psi3 = vel_short(nD,nR,DD,RRd)
        return psi3
    elif (estimator  == "turner"):
        psi3 = turner(nD,nR,DD,DR,RD,RRn,RRd)
        return psi3
    else:
        return "psi3 estimator must be either 'short' or 'turner'"

def calc_xiGG(nD,nR,DD,DR,RR,estimator = "landy_szalay"):
    if (estimator == "peebles"):
        xiGG = peebles(nD,nR,DD,RR)
        return xiGG
    elif (estimator  == "landy_szalay"):
        xiGG = landy_szalay(nD,nR,DD,DR,RR)
        return xiGG
    else:
        return "psi3 estimator must be either 'peebles' or 'landy_szalay'" 
    
# Multipole calculation for \xi_gu and \xi_gg
# xi_gg should be pretty much identical to this, but without the (1/2) factor in the actual calculation
# and a 'folding over' of the matrix/array to account for the fact that the integral actually starts at 0, not -1
def multipole_psi3(data, ell, del_mu, sBins):
    # I'm choosing the bounds of arange to be the midpoints of the bins
    range_start = -1.0 + del_mu/2.
    range_end = -1.*range_start
    # sum_range should haave length equal to the number of bins used for cos(theta_mu)
    muBins = int(2.0/del_mu)
    sum_range = np.linspace(range_start, range_end, muBins)
    # reshape the data now into a 2D (s,mu) matrix
    data = (data * del_mu * ((ell*2.+1.)/2.)).reshape((sBins,muBins),order="C")
    # data = data.reshape((sBins, muBins))
    if(int(ell) == 1):
        l1_range = sum_range
        multipole_data = data * l1_range
    elif(int(ell) == 3):
        l3_range = 0.5*(5.*sum_range**3. - 3.*sum_range)
        multipole_data = data * l3_range
    else: raise Exception("Please ask for a non-zero multipole of the cross-correlation ( dipole (1) or octupole (3) )!")
    return np.nansum(multipole_data,axis=1)

def multipole_xigg(data, ell, del_mu, sBins):
    muBins = int(2.0/del_mu)
    if(muBins % 2 != 0):
        raise Exception("Number of mu bins for gg multipoles should be even (del_mu = 0.05, 0.1, 0.2, etc...).")
    range_start = -1.0 + del_mu/2.0
    range_end = 1.0 - del_mu/2.0
    sum_range = np.linspace(range_start, range_end, muBins)
    data = (data * del_mu * ((ell*2.0+1.0))).reshape((sBins,muBins),order="C")
    if(int(ell) == 0):
        multipole_data = data
    elif(int(ell) == 2):
        l2_range = 0.5*(3.*sum_range**2. - 1.0)
        multipole_data = data * l2_range
    elif(int(ell) == 4):
        l4_range = 0.125*(35.*sum_range**4. - 30.*sum_range**2. + 3.0)
        multipole_data = data * l4_range
    else: raise Exception("Please ask for a non-zero multipole of the 2PCF ( monopole (0), quadrupole (2) or hexadecapole (4) )!")
    datlen = data.shape[1]
    halflen = int(datlen/2)
    sumdata = np.zeros((sBins,halflen))
    i = 0
    while i < range.size/2: 
        sumdata[:,(halflen - (i+1))] = multipole_data[:,i] + multipole_data[:,(datlen - (i+1))]
        i += 1
    return np.sum(sumdata, axis=1)
