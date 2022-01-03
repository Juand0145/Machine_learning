#!usr/bin/env python3
"""File that contains the function one_hot_decode"""


def one_hot_decode(one_hot):
    import numpy as np
    """
    Function that converts a one-hot matrix into a vector of labels
    Args:
    one_hot: is a one-hot encoded numpy.ndarray with shape (classes, m)
        classes is the maximum number of classes
        m is the number of examples
    Returns: a numpy.ndarray with shape (m, ) containing the numeric
    labels for each example, or None on failure
    """
    one_hot = one_hot.T

    decode = []
    rows = one_hot.shape[0]
    columns = one_hot.shape[1]

    for i in range(rows):
        for j in range(columns):
            if one_hot[i][j] == 1:
                decode.append(j)

    decode = np.array(decode)

    return decode
