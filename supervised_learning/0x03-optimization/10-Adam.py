#!/usr/bin/env python3
"""File that contains the function create_Adam_op"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a neural
    network in tensorflow using the Adam optimization algorithm
    Args:
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(alpha,
                                       beta1,
                                       beta2,
                                       epsilon).minimize(loss)

    return optimizer
