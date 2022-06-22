#!/usr/bin/env python3
"""File that contains the function epsilon_greedy"""
import numpy as np
import gym.envs.toy_text.frozen_lake as frozen_lake


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action
    Args:
      Q is a numpy.ndarray containing the q-table
      state is the current state
      epsilon is the epsilon to use for the calculation
      You should sample p with numpy.random.uniformn to determine if
      your algorithm should explore or exploit
      If exploring, you should pick the next action with numpy.random.randint
      from all possible actions
    Returns: the next action index
    """
    e_tradeoff = np.random.uniform(0, 1)

    if e_tradeoff < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])

    return action
