#!/usr/bin/env python3
"""File that contains the function create_batch_norm_layer"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that  creates a batch normalization layer for a neural
    network in tensorflow
    Args:
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output
    of the layer
    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n, kernel_initializer=init, name='layer')
    epsilon = 1e-8

    base = layer(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    mean, variance = tf.nn.moments(base, axes=[0])
    Z = tf.nn.batch_normalization(base, mean=mean,
                                  variance=variance,
                                  offset=beta,
                                  scale=gamma,
                                  variance_epsilon=epsilon)
    return activation(Z)
