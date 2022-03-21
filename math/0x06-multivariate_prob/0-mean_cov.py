#!/usr/bin/env python3
"""File that contains the function mean_cov"""
import numpy as np


def mean_cov(X):
    """
    function that calculates the mean and covariance of a data set
     is a numpy.ndarray of shape (n, d) containing the data set:
    n is the number of data points
    d is the number of dimensions in each data point
    If X is not a 2D numpy.ndarray, raise a TypeError with the
    message X must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the message X must
    contain multiple data points
    Returns: mean, cov:
    mean is a numpy.ndarray of shape (1, d) containing the mean of the
    data set
    cov is a numpy.ndarray of shape (d, d) containing the covariance
    matrix of the data set
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = np.mean(X, axis=0).reshape(1, d)
    X_mean = X - mean

    cov = np.matmul(X_mean.T, X_mean) / (n - 1)

    return mean, cov
