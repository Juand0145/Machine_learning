#!/usr/bin/env python3
"""File that contains the function grads"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Function that calculates the gradients of Y
    Args:
     is a numpy.ndarray of shape (n, ndim) containing the
     low dimensional transformation of X
    P is a numpy.ndarray of shape (n, n) containing the P
    affinities of X
    Do not multiply the gradients by the scalar 4 as described
    in the paper’s equation
    Returns: (dY, Q)
    dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    You may use Q_affinities = __import__('5-Q_affinities').Q_affinities
    """

    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))

    PQ = P - Q
    PQ_expanded = np.expand_dims((PQ * num).T, axis=2)

    for i in range(n):
        y_diff = Y[i, :] - Y
        dY[i, :] = np.sum((PQ_expanded[i, :] * y_diff), 0)

    return dY, Q
