#!/usr/bin/env python3


def minor(matrix):
    """
    Function that calculates the minor of a mtarix
    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix
    """
    determinant = __import__("0-determinant").determinant

    if matrix == [[]]:
        raise ValueError("matrix must be a square matrix")

    try:
        flag = matrix[0][0]
    except:
        raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 1:
        return [[1]]

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    row = len(matrix)

    a = []
    result = []

    for x in range(row):
        for y in range(row):
            for i in range(row):
                for j in range(row):
                    if i != x and j != y:
                        a.append(matrix[i][j])

            new_matrix = splice(a, row - 1)
            a = []

            minor_value = determinant(new_matrix)
            result.append(minor_value)

    return (splice(result, row))


def splice(array, splitedSize):
    """
    Function that tranform an array into a square matrix:
    array: the array to transform
    splitedSize: the size of the matrix
    return a square matrix
    """
    spliced_matrix = [array[x:x+splitedSize]
                      for x in range(0, len(array), splitedSize)]
    return spliced_matrix
