#!/usr/bin/env python3
"""Is a function that adds two matrices"""


def add_matrices(mat1, mat2):
    """Function that adds two matrices"""
    import numpy as np

    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    if mat1.shape != mat2.shape:
        return None

    result = np.add(mat1, mat2).tolist()

    return result
