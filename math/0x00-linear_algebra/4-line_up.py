#!/usr/bin/env python3
"""Is a function that adds two arrays element-wise"""

def add_arrays(arr1, arr2):
    """Dunction that adds two arrays element-wise"""

    if len(arr1) != len(arr2):
        return None

    arr3 = []
    for i in range(len(arr1)):
        sum = arr1[i] + arr2[i]
        arr3.append(sum)

    return arr3