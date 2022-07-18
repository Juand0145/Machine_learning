#!/usr/bin/env python3
"""File that contains the function from_dictionary"""
import pandas as pd


def from_dictionary():
    """
    Function that creates a pd.DataFrame from a dictionary
    Args:
      The first column should be labeled First and have the values
      0.0, 0.5, 1.0, and 1.5
      The second column should be labeled Second and have the values
      one, two, three, four
      The rows should be labeled A, B, C, and D, respectively
      The pd.DataFrame should be saved into the variable df
    """
    df = pd.DataFrame(
        {
            "First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]
        },
        index=list("ABCD"))
    return df


df = from_dictionary()
