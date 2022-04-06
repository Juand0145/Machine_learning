#!/usr/bin/env python3
"""File that contains the function absorbing"""
import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities of a regular markov
    chain
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n) representing the
        standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    try:
        if len(P.shape) != 2:
            return None
        n = P.shape[0]
        if n != P.shape[1]:
            return None

        #  (πP).T = π.T ⟹ P.T π.T = π.T (.)
        evals, evecs = np.linalg.eig(P.T)

        # trick: has to be normalized
        state = (evecs / evecs.sum())

        # P.T π.T = π.T (.)
        new_state = np.dot(state.T, P)
        for i in new_state:
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)
    except Exception:
        return None
