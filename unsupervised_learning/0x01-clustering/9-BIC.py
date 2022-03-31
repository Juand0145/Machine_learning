#!/usr/bin/env python3
"""File that contains the function BIC"""
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that  finds the best number of clusters for a GMM using the
    Bayesian Information Criterion
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    expectation_maximization = __import__('8-EM').expectation_maximization

    best_k = np.arange(kmin, kmax + 1)
    best_result = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S,  _, log_l = expectation_maximization(X, k, iterations, tol,
                                                       verbose)
        best_k.append(k)
        best_result.append((pi, m, S))
        logl_val.append(log_l)

        # Formula p :https://bit.ly/33Cw8lH
        cov_params = d * (d + 1) / 2.
        mean_params = d
        p = int((k * cov_params) + (k * mean_params) + k - 1)

        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    best_k = best_k[best_val]
    best_result = best_result[best_val]

    return best_k, best_result, logl_val, bic
