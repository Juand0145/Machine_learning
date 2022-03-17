#!/usr/bin/env python3
"""File that contains the function definiteness"""


def definiteness(matrix):
    """
    matrix is a numpy.ndarray of shape (n, n) whose definiteness
    should be calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the
    message matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None
    Return: the string Positive definite, Positive semi-definite, Negative
    semi-definite, Negative definite, or Indefinite if the matrix is positive
    definite, positive semi-definite, negative semi-definite, negative
    definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    You may import numpy as np
    """
    import numpy as np

    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.all(matrix.T == matrix):
        return None

    w, v = np.linalg.eig(matrix)

    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
