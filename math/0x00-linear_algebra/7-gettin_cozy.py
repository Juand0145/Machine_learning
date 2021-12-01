#!/usr/bin/env python3
"""Is a funtion that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """funtion that concatenates two matrices along a specific axis"""
    import numpy as np

    matrix_1 = np.matrix(mat1)
    matrix_2 = np.matrix(mat2)

    matrix_3 = np.concatenate((matrix_1, matrix_2), axis).tolist()

    return matrix_3
