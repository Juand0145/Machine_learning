#!/usr/bin/env python3
"""Fila that contains the function create_train_op"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network
    Args:
    loss: is the loss of the network’s prediction
    alpha: is the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    gradient_decent = tf.train. GradientDescentOptimizer(alpha).minimize(loss)

    return gradient_decent
