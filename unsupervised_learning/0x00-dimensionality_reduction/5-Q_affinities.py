#!/usr/bin/env python3
"""File that contains the function Q_affinities"""
import numpy as np


def Q_affinities(Y):
    """
    Function that calculates the Q affinities
    Args:
    Y is a numpy.ndarray of shape (n, ndim) containing the low
    dimensional transformation of X
    n is the number of points
    ndim is the new dimensional representation of X
    Returns: Q, num
    Q is a numpy.ndarray of shape (n, n) containing
    the Q affinities
    num is a numpy.ndarray of shape (n, n) containing
    the numerator of the Q affinities
    """
    sum_Y = np.sum(np.square(Y), 1)
    numerator = -2. * np.dot(Y, Y.T)
    numerator = 1. / (1. + np.add(np.add(numerator, sum_Y).T, sum_Y))
    np.fill_diagonal(numerator, 0)
    Q = numerator / np.sum(numerator)

    return Q, numerator
