#!/usr/bin/env python3
"""File that contains the function l2_reg_cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Function that calculates  the cost of a neural network
    with L2 regularization:
    Args:
    cost is the cost of the network without L2 regularization
    lambtha is the regularization parameter
    weights is a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network
    L is the number of layers in the neural network
    m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    total = 0

    for key, values in weights.items():
        if key[0] == 'W':
            total = total + np.linalg.norm(values)

    cost_l2 = cost + ((lambtha * total) / (2 * m))
    return cost_l2
