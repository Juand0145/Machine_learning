#!/usr/bin/env python3
"""Is a function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""

    try:
        iter(matrix)
        elements = len(matrix)
        return [elements] + matrix_shape(matrix[0])

    except:
        return [elements]