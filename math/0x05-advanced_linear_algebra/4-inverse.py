#!/usr/bin/env python3
"""File that contains the function inverse"""


def inverse(matrix):
    """
    matrix is a list of lists whose inverse should be calculated
    If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with
    the message matrix must be a non-empty square matrix
    Returns: the inverse of matrix, or None if matrix is singular
    """
    determinant = __import__("0-determinant").determinant
    cofactor = __import__("2-cofactor").cofactor
    transpose = __import__("3-adjugate").matrix_transpose

    row = len(matrix)

    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all([len(mat) == row for mat in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if determinant(matrix) == 0:
        return None
    if len(matrix) == 1:
        return [[1/matrix[0][0]]]

    cofactor_matrix = cofactor(matrix)

    inverse_matrix = transpose(cofactor_matrix)

    for i, rows in enumerate(inverse_matrix):
        for j, values in enumerate(rows):
            inverse_matrix[i][j] = values / determinant(matrix)

    return inverse_matrix
