#!/usr/bin/env python3
"""Is a function that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    import numpy as np

    try:
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)

        result = np.concatenate((mat1, mat2)).tolist()
        return result

    except:
        return None
