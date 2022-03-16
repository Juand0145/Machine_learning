#!/usr/bin/env python3
"""File that contains the function """


def adjugate(matrix):
    """
    Function that calculates the adjugate matrix of a matrix
    matrix is a list of lists whose adjugate matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the
    message matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix
    """
    cofactor = __import__("2-cofactor").cofactor
    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = matrix_transpose(cofactor_matrix)

    return adjugate_matrix


def matrix_transpose(matrix):
    """
    Function that returns the transpose of a 2D matrix
    matrix: A list of list that represent a mtrix
    return the transpose of the matrix
    """
    tranpose = []
    row = []

    for i in range(len(matrix[0])):
        for j in matrix:
            row.append(j[i])

        tranpose.append(row)
        row = []

    return tranpose
