#!/usr/bin/env python3
"""File that conains the function kmeans"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Function that performs K-means on a dataset
    Args:
    X is a numpy.ndarray of shape (n, d) containing the dataset
      n is the number of data points
      d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed.
    Returns: C, clss, or None, None on failure
      C is a numpy.ndarray of shape (k, d) containing the centroid means for
      each cluster.
      clss is a numpy.ndarray of shape (n,) containing the index of the cluster
      in C that each data point belongs to.
    """
    initialize = __import__("0-initialize").initialize
    if type(iterations) is not int or iterations <= 0:
        return None, None

    C = initialize(X, k)

    if C is None:
        return None, None

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    n, d = X.shape

    for i in range(iterations):
        centroids = np.copy(C)
        centroids_ext = C[:, np.newaxis]

        distances = np.sqrt(((X - centroids_ext) ** 2).sum(axis=2))

        clss = np.argmin(distances, axis=0)

        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis=0)

        centroids_ext = C[:, np.newaxis]
        distances = np.sqrt(((X - centroids_ext) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        if (centroids == C).all():
            break

    return C, clss
