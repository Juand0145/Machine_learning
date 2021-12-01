#!/usr/bin/env python3
"""Is a function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    elements = len(matrix)

    if type(matrix[0]) is list:
        return [elements] + matrix_shape(matrix[0])

    return [elements]
