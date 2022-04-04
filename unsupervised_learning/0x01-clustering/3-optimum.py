#!/usr/bin/env python3
"""File that contains the function optimum_k"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    File that tests for the optimum number of clusters by variance
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters
    to check
    for (inclusive)
    kmax is a positive integer containing the maximum number of clusters
    to check
    for (inclusive)
    iterations is a positive integer containing the maximum number of
    iterations for K-means
    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means for each cluster
        size
        d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size
    """
    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance
    try:
        if type(X) is not np.ndarray or len(X.shape) != 2:
            return None, None
        if type(iterations) != int or iterations < 1:
            return None, None
        if kmax is not None and (type(kmax) is not int or kmax < 1):
            return None, None
        if kmax is not None and kmin >= kmax:
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if not isinstance(kmin, int) or kmin < 1 or kmin >= X.shape[0]:
            return None, None
        
        

        results = []
        var = []

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))

            var.append(variance(X, C))

        d_var = [0]
        for i in range(kmax - 1):
            derivate = var[i] - var[i + 1]
            d_var.append(derivate + d_var[i])

        return results, d_var

    except Exception:
        return None, None
