import gc
import math
import numpy as np
import dolfin as dl
from scipy.linalg import eigh, svd, solve
from scipy.fftpack import fft, ifft, ifftshift

def autocorrelation(x: np.ndarray) -> np.ndarray:
    """
    :param x: Time series
    :return: Autocorrelation of the time series
    This function is grabbed from stack overflow
    """
    xp = ifftshift((x - np.average(x)) / np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n // 2], np.zeros_like(xp), xp[n // 2:]]
    f = fft(xp)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    return np.real(pi)[:n // 2] / (np.arange(n // 2)[::-1] + n // 2)

def interagted_autocorrelation_time(accor: np.ndarray) -> np.ndarray:
    """
    :param accor: autocorrelation function
    :return: 1. Integrated autocorrelation time.
             2. The time (truncated at which the nearby autocorrelation sum is negative).
    """
    nparam = accor.shape[0] # number of parameters
    time = accor.shape[1] # number of time steps
    npair = int(math.floor(0.5 * time)) # number of pairs
    iat = np.zeros(nparam) # integrated autocorrelation time
    ttime = np.zeros(nparam).astype("int") # truncated time
    P = np.zeros((nparam, npair)) # autocorrelation neighbor sum
    for i in range(npair): # compute the autocorrelation sum
        P[:, i] = accor[:, 2 * i] + accor[:, 2 * i + 1]  # compute the neighbor sum
    for i in range(nparam): #loop over the parameters
        idx = np.where(P[i, :] <= 0)[0] # find the first negative sum
        if idx.size == 0: # if no negative sum
            ttime[i] = npair # truncated time is the maximum
        else: # if there is a negative sum
            ttime[i] = idx[0] # truncated time is the first negative sum
        iat[i] = -1 + 2 * np.sum(P[i, :ttime[i]]) # compute the integrated autocorrelation time
    return iat, 2 * ttime # return the integrated autocorrelation time and the truncated time

def SingleChainESS(samples):
    """
    :param samples: The samples of the Markov chain. Has shape (n_samples, n_parameters)
    :return: The effective sample size of size (n_parameters,)
    """
    for ii in range(samples.shape[1]):
        if ii == 0:
            temp = autocorrelation(samples[:, ii])
            ntime = temp.size
            acorr = np.zeros((samples.shape[1], ntime))
            acorr[0, :] = temp
        else:
            acorr[ii, :] = autocorrelation(samples[:, ii])
    iat, _ = interagted_autocorrelation_time(acorr)
    return 1.0 / iat * 100