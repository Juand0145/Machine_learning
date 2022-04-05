#!/usr/bin/env python3
"""File that contains the function viterbi"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Function that calculates the most likely sequence of hidden states for a
    hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the
            hidden state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state
    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward
        path probabilities
        F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
    """
    try:
        T = Observation.shape[0]
        N, M = Emission.shape

        backpointer = np.zeros((N, T))

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.amax(Transitions * F[:, t - 1]
                                  * Emissions)
                backpointer[n, t - 1] = np.argmax(Transitions * F[:, t - 1]
                                                  * Emissions)

        path = [0 for i in range(T)]

        last_state = np.argmax(F[:, T - 1])
        path[0] = last_state

        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            path[backtrack_index] = int(backpointer[int(last_state), i])
            last_state = backpointer[int(last_state), i]
            backtrack_index += 1

        path.reverse()

        P = np.amax(F[:, T - 1], axis=0)

        return path, P
    except Exception:
        None, None
