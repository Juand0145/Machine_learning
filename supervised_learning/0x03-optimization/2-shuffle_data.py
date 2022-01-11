#!/usr/bin/env python3
"""File that contains the function shuffle_data"""
import numpy as np
import tensorflow.compat.v1 as tf


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way
    Args:
    X is the first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    Returns: the shuffled X and Y matrices
    """
    X = np.random.permutation(X)
    Y = np.random.permutation(Y)

    return X, Y