#!/usr/bin/env python3
"""Is a function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """Function that returns the transpose of a 2D matrix"""

    tranpose = []
    row =[]    

    for i in range(len(matrix[0])):
        for j in matrix:
            row.append(j[i])

        tranpose.append(row)
        row =[]

    return tranpose