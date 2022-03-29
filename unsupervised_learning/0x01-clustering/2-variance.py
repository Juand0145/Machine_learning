#!/usr/bin/env python3
"""File that contains the function variance"""
import numpy as np


def variance(X, C):
    """
    File that calculates the total intra-cluster variance for a data set
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    Returns: var, or None on failure
        var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None

    n, d = X.shape
    k, d = C.shape

    centroids_ext = C[:, np.newaxis]
    distances = np.sqrt(((X - centroids_ext)**2).sum(axis=2))
    min_distances = np.min(distances, axis=0)

    variance = np.sum(min_distances ** 2)

    return variance
