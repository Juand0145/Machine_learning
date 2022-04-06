#!/usr/bin/env python3
"""File that contains the function markov_chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain being in
    a particular state after a specified number of iterations
    Args:
        P is a square 2D numpy.ndarray of shape (n, n) representing the
        transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
        s is a numpy.ndarray of shape (1, n) representing the probability of
        starting in each state
        t is the number of iterations that the markov chain has been through
    Returns: a numpy.ndarray of shape (1, n) representing the probability of
    being in a specific state after t iterations, or None on failure
    """
    if len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if (n1 != n2) or type(P) is not np.ndarray or not isinstance(t, int):
        return None
    if t < 0:
        return None
    if n1 != s.shape[1] or s.shape[0] != 1:
        return None

    S = np.matmul(s, P)

    for i in range(t):
        S = np.matmul(S, P)

    return S
