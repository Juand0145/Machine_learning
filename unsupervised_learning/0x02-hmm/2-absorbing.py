#!/usr/bin/env python3
"""File that contains the function absorbing"""
import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n) representing the
        standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
        Returns: True if it is absorbing, or False on failure
    """
    if len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if (n1 != n2) or type(P) is not np.ndarray:
        return None
    D = np.diagonal(P)
    if (D == 1).all():
        return True
    if not (D == 1).any():
        return False

    for i in range(n1):
        for j in range(n2):
            if (i == j) and (i + 1 < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False
    return True
