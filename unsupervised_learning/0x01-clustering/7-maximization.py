#!/usr/bin/env python3
"""File that contains the function maximization"""
import numpy as np


def maximization(X, g):
    """
    File that calculates the maximization step in the EM algorithm for a GMM
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors
        for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.array:
        return None, None, None

    n, d = X.shape
    k, n = g.shape

    pi = np.sum(g, axis=1) / n

    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m_nume = np.matmul(g[i], X)
        m_deno = np.sum(g[i])
        m[i] = m_nume/m_deno

        centroids = X - m[i]
        S_num = np.matmul(g[i] * centroids.T, centroids)
        S[i] = S_num/m_deno

    return pi, m, S
