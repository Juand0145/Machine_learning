#!/usr/bin/env python3
"""File that contains the function initialize"""
import numpy as np


def initialize(X, k):
    """
    File that that initializes variables for a Gaussian Mixture Model
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors for each
    cluster, initialized evenly
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster, initialized with K-means
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for each cluster, initialized as identity matrices
    """
    kmeans = __import__('1-kmeans').kmeans

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k < 1:
        return None, None, None

    n, d = X.shape

    pi = np.ones(k)/k

    m, _ = kmeans(X, k)

    sigma_matrix = np.tile(np.identity(d), (k, 1))
    S = sigma_matrix.reshape(k, d, d)

    return pi, m, S
