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
    if np.all(P) != 1:
        return True

    else:
        return False
