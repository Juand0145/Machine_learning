#!/usr/bin/env python3
"""File that contains the function P_init"""
import numpy as np


def P_init(X, perplexity):
    """
    Function that nitializes all variables required to calculate
    the P affinities in t-SNE:
    Args:
    X is a numpy.ndarray of shape (n, d) containing the dataset to
    be transformed by t-SNE
    n is the number of data points
    d is the number of dimensions in each point
    perplexity is the perplexity that all Gaussian distributions should have
    Returns: (D, P, betas, H)
    D: a numpy.ndarray of shape (n, n) that calculates the squared pairwise
    distance between two data points
    The diagonal of D should be 0s
    P: a numpy.ndarray of shape (n, n) initialized to all 0s that will contain
    the P affinities
    betas: a numpy.ndarray of shape (n, 1) initialized to all 1s that will
    contain all of the beta values
    \beta_{i} = \frac{1}{2{\sigma_{i}}^{2} }
    H is the Shannon entropy for perplexity perplexity with a base of 2
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)

    D = (np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X))
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
