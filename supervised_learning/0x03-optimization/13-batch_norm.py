#!/usr/bin/env python3
"""File that contains the function """


def batch_norm(Z, gamma, beta, epsilon):
    """
    function that normalizes an unactivated output of a neural network
    using batch normalization:
    Args:
    Z is a numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n) containing the scales used for batch normalization
    beta is a numpy.ndarray of shape (1, n) containing the offsets used for batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Z_normalized = (Z - mean) / variance

    gamma = pow(variance + epsilon, 0.5)
    beta = mean
    Z_batch_normalized = gamma * Z_normalized + beta

    return Z_batch_normalized
