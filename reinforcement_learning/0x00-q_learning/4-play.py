#!/usr/bin/env python3
"""File that contains the function play"""
import numpy as np
import gym.envs.toy_text.frozen_lake as frozen_lake


def play(env, Q, max_steps=100):
    """
    Function that has the trained agent play an episode
    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        max_steps is the maximum number of steps in the episode
        Each state of the board should be displayed via the console
        You should always exploit the Q-table
    Returns: the total rewards for the episode
    """
    # reset the state
    state = env.reset()
    env.render()
    done = False

    for step in range(max_steps):
        # take the action with maximum expected future reward form the q-table
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)

        if done is True:
            env.render()
            return reward
        env.render()
        state = new_state

    # close the connection to the environment
    env.close()
    return reward
