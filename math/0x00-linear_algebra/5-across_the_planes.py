#!/usr/bin/env python3
"""Is a function hat adds two matrices element-wis"""


def add_matrices2D(mat1, mat2):
    """Function hat adds two matrices element-wis"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    row = []
    new_matrix = []

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])

        new_matrix.append(row)
        row = []

    return new_matrix
