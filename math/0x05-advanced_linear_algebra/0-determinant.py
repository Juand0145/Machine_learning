#!/usr/bin/env python3
"""File that contains the function determinant"""


def determinant(matrix):
    """
    Function that calculates the determinant from a matrix:
    matrix: is a list of lists whose determinant should be calculated
        If matrix is not a list of lists, raise a TypeError with the
        message matrix must be a list of lists
        If matrix is not square, raise a ValueError with the message
        matrix must be a square matrix
    The list [[]] represents a 0x0 matrix
    Returns: the determinant of matrix
    """
    if matrix == [[]]:
        return 1

    try:
        flag = matrix[0][0]
    except Exception:
        raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 1:
        return (matrix[0][0])

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    return deter(matrix)


def deter(matrix):
    """
    Function that calculates the determinant from a matrix
    matrix: is a list of lists whose determinant should be calculated
    """
    row = len(matrix)

    if row == 2:
        two_two = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return two_two

    else:
        a = []
        result = []

        for x in range(row):

            for i in range(row):
                for j in range(row):
                    if i != 0 and j != x:
                        a.append(matrix[i][j])

            splitedSize = row - 1
            new_matrix = [a[x:x+splitedSize]
                          for x in range(0, len(a), splitedSize)]

            if x % 2:
                result.append(matrix[0][x] * -deter(new_matrix))
            else:
                result.append(matrix[0][x] * +deter(new_matrix))
            a = []

        return (sum(result))
