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

        # note: the matrix is row stochastic.
        # A markov chain transition will correspond to left multiplying
        # by a row vector.
        Q = P

        # We have to transpose so that Markov transitions correspond to right
        # multiplying by a column vector. np.linalg.eig finds right
        # eigenvectors.
        evals, evecs = np.linalg.eig(Q.T)
        evec1 = evecs[:, np.isclose(evals, 1)]

        # Since np.isclose will return an array, we've indexed with an array
        # so we still have our 2nd axis. Get rid of it, since it's only size 1.
        evec1 = evec1[:, 0]

        stationary = evec1 / evec1.sum()
        if np.sum(stationary) != 1:
            return None
        if np.all(P) != 1:
            return None

        return(stationary)

    except Exception:
        return None
