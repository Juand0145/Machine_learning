#!/usr/bin/env python3
"""File that contains the function pdf"""
import numpy as np


def pdf(X, m, S):
    """
    Function that calculates the probability density function of a Gaussian
    distribution
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
    should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
    distribution
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for
        each data point
        All values in P should have a minimum value of 1e-300
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    P = np.zeros(n)
    for i in range(n):

        pi = (2 * np.pi) ** (d/2)
        sigma_1 = np.linalg.det(S) ** (1/2)

        var_1 = (X[i] - m, )
        sigma_2 = np.linalg.inv(S)
        var_2 = np.transpose((X[i] - m, ))

        equation_1 = 1/(pi * sigma_1)
        equation_2 = np.matmul(np.matmul(var_1, sigma_2), var_2)

        PDF = equation_1 * np.exp(- (1/2) * equation_2)
        P[i] = np.squeeze(PDF)

    P = np.where(P < 1e-300, 1e-300, P)

    return P
