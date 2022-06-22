#!/usr/bin/env python3
"""File that contains the function load_frozen_lake"""
import numpy as np
import gym.envs.toy_text.frozen_lake as frozen_lake


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Function that loads the pre-made FrozenLakeEnv evnironment from
    OpenAI’s gym
    Args:
      desc is either None or a list of lists containing a custom description
      of the map to load for the environment
      map_name is either None or a string containing the pre-made map to load
      Note: If both desc and map_name are None, the environment will load a
      randomly generated 8x8 map
      is_slippery is a boolean to determine if the ice is slippery
    Returns: the environment
    """
    env = frozen_lake.FrozenLakeEnv(desc,
                                    map_name,
                                    is_slippery)

    return env
