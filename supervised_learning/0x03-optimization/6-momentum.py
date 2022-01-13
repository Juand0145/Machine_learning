#!/usr/bin/env python3
"""File that contains the function create_momentum_o"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Funtion that updates a variable using the gradient descent with
    momentum optimization algorithm:
    Args:
    alpha is the learning rate
    beta1 is the momentum weight
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    Returns: the updated variable and the new moment, respectively
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

    return optimizer
