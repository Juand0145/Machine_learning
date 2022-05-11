#!/usr/bin/env python3
"""File that conatins teh function preprocess_data"""

def preprocess_data(data_frame):
  """
  Function that remove the rows with empty values
  Args:
    data_frame: Is a numpy.dataframe with cripto's price information
  Return: the new_data_fram filtered
  """
  columns = data_frame.columns
  NaN = data_frame[columns[1]] > 0
  new_data_frame = data_frame.loc[NaN]

  return new_data_frame
