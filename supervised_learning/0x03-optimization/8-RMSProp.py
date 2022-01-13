#!/usr/bin/env python3
"""File that conatains the function create_RMSProp_op"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Function that  creates the training operation for
    a neural network in tensorflow using the RMSProp
    optimization algorithm
    Args:
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon).minimize(loss)

    return optimizer
