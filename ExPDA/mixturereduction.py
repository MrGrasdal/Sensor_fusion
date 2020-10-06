from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    mean: np.ndarray,  # the mixture means shape(N, n)
    cov: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""
    
    # mean
    mean_bar = np.average(mean, 0, w)   
    # covariance

    # # internal covariance
    cov_int = np.average(cov, 0, w)

    # # spread of means
    cov_ext = np.average(mean * mean, 0, w) - mean_bar * mean_bar
    # # total covariance
    cov_bar = cov_int + cov_ext

    return mean_bar, cov_bar
    