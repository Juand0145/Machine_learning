#!/usr/bin/env python3
"""File that contains the function"""
import numpy as np


def expectation(X, pi, m, S):
    """
    File that calculates the expectation step in the EM algorithm for a GMM
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for each
    cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for each cluster
    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
        l is the total log likelihood
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    pdf = __import__('5-pdf').pdf

    n, d = X.shape
    k = pi.shape[0]

    g = np.zeros([k, n])
    denominator = 0
    for i in range(k):
        likelihood = pdf(X, m[i], S[i])

        numerator = likelihood * pi[i]
        g[i] = numerator

        denominator += numerator

    log_likelihood = np.sum(np.log(np.sum(g, axis=0)))
    g = g / denominator

    return g, log_likelihood
