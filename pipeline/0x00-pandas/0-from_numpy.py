#!/usr/bin/env python3
"""File tha contains the function from_numpy"""
import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray
    Args:
        array is the np.ndarray from which you should create the pd.DataFrame
        The columns of the pd.DataFrame should be labeled in alphabetical
        order and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]
    number_columns = (array.shape)[1]
    columns_names = [alphabet[x] for x in range(number_columns)]

    df = pd.DataFrame(array, columns=columns_names)

    return df
