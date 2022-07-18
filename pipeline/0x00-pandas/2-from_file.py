#!/usr/bin/env python3
"""File that contains the function from_file"""
import numpy as pd


def from_file(filename, delimiter):
    """
    Function that loads data from a file as a pd.DataFrame
    Args:
      filename is the file to load from
      delimiter is the column separator
    Returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
