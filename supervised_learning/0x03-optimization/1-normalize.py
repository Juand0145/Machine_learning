#!/usr/bin/env python3
"""File that conatins the function normalize"""
import numpy as np
import tensorflow.compat.v1 as tf


def normalize(X, m, s):
    """
    X is the numpy.ndarray of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m is a numpy.ndarray of shape (nx,) that contains the mean
    of all features of X
    s is a numpy.ndarray of shape (nx,) that contains the standard
    deviation of all features of X
    Returns: The normalized X matrix
    """
    X = (X-m)/s

    return X
