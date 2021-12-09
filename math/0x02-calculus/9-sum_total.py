#!/usr/bin/env python3
"""Is a function  that calculate the sumatory of i^2"""


def summation_i_squared(n):
    """Function  that calculate the sumatory of i^2"""
    import numpy as np

    if type(n) is not int or n <= 0:
        return None

    array = np.arange(n + 1)
    result = np.sum(array**2)

    return result
