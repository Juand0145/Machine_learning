#!/usr/bin/env python3
"""File that contains the function create_layer"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Args:
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    """
    initialize = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initialize, name="layer")

    new_layer = layer(prev)

    return new_layer
