#!/usr/bin/env python3
"""Is a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication"""
    import numpy as np

    columns = len(mat1[0])
    rows = len(mat2)

    if columns != rows:
        return None

    mat1 = np.matrix(mat1)
    mat2 = np.matrix(mat2)

    mat3 = np.matmul(mat1, mat2).tolist()

    return mat3
