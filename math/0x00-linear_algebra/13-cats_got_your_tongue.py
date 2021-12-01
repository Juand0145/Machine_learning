#!/usr/bin/env python3
"""Is a function that concatenates two matrices along a specific axis"""


def np_cat(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    import numpy as np

    matrix_3 = np.concatenate((mat1, mat2), axis)

    return matrix_3
