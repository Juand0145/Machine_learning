#!/usr/bin/env python3
"""File that contains the function q_init"""
import numpy as np
import gym.envs.toy_text.frozen_lake as frozen_lake


def q_init(env):
    """
    Function that  initializes the Q-table
    Args:
      env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    env_shape, _ = env.desc.shape
    Q_rows = env_shape ** 2
    Q_table = np.zeros((Q_rows, 4))

    return(Q_table)
