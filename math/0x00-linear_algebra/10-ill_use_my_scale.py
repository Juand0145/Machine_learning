#!/usr/bin/env python3
"""Is a function that calculates the shape of a numpy.ndarray"""


def np_shape(matrix):
    """Function that calculates the shape of a numpy.ndarray"""
    import numpy as np

    shape = np.shape(matrix)

    return(shape)
