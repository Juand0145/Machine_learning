#!/usr/bin/env python3
"""File that contains the function dropout_create_layer"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Function that creates a layer of a neural network using dropout
    Args:
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on the layer
    keep_prob is the probability that a node will be kept
    Returns: the output of the new layer
    """

    dropout = tf.keras.layers.Dropout(keep_prob)
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg"))

    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=initializer,
                             kernel_regularizer=dropout)

    output = tensor(prev)

    return output
