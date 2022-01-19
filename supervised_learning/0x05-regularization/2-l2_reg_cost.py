#!/usr/bin/env python3
"""File hat contains the funtion l2_reg_cost"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Function that  calculates the cost of a neural network with
    L2 regularization
    Args:
    cost is a tensor containing the cost of the network without L2
    regularization
    Returns: a tensor containing the cost of the network accounting
    for L2 regularization
    """
    regularized_cost = cost + tf.losses.get_regularization_losses()

    return regularized_cost
