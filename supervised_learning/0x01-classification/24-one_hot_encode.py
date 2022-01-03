#!/usr/bin/env python3
"""File that contains the one_hot_encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Function that converts a numeric label vector
    into a one-hot matrix
    Args:
    Y: is a numpy.ndarray with shape (m,) containing numeric
    class labels
        m is the number of examples
    classes: is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m),
    or None on failure
    """
    try:
        b = np.zeros((Y.size, classes))

        b[np.arange(Y.size), Y] = 1

        return b.T

    except:
        return None
