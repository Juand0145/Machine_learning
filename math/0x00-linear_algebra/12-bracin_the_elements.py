#!/usr/bin/env python3
"""Is a function that performs element-wise addition, subtraction,
multiplication, and division:"""
import numpy as np


def np_elementwise(mat1, mat2):
    """Function that performs element-wise addition, subtraction,
    multiplication, and division"""

    add = np.add(mat1, mat2)

    sub = np.subtract(mat1, mat2)

    mul = np.multiply(mat1, mat2)

    div = np.divide(mat1, mat2)

    return add, sub, mul, div
