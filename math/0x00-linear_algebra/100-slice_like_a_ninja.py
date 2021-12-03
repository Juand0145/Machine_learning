#!/usr/bin/env python3
"""Is a function that slices a matrix along a specific axes"""


def np_slice(matrix, axes={}):
    """Function that slices a matrix along a specific axes"""
    piece = (slice(*axes.get(depth, (None, None)))
             for depth in range(len(matrix.shape)))

    sliced_matrix = matrix[tuple(piece)]

    return sliced_matrix
