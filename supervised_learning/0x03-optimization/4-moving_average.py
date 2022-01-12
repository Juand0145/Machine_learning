#!/usr/bin/env python3
"""File that contains he function moving_average"""


def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set
    Args:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """

    Vt = 0
    EWMA = []

    for i in range(len(data)):
        Vt = (beta * Vt) + ((1 - beta) * data[i])
        bias_correction = 1 - beta ** (i + 1)
        new_Vt = Vt / bias_correction

        EWMA.append(new_Vt)
    return EWMA
