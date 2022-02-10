#!/usr/bin/env python3
"""File that contain the function dense_block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that  builds a dense block as described in Densely
    Connected Convolutional Networks
    Args:
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively
    """
    initializer = K.initializers.he_normal
    function = "relu"

    normalization_1 = K.layers.BatchNormalization()(X)

    function_1 = K.layers.Activation(function)(normalization_1)

    bottleneck = K.layers.Conv2D(filters=4*growth_rate,
                                 kernel_size=1,
                                 padding="same",
                                 kernel_initializer=initializer)(function_1)

    normalization_2 = K.layers.BatchNormalization()(bottleneck)

    function_2 = K.layers.Activation(function)(normalization_2)

    X_convolution = K.layers.Conv2D(filters=growth_rate,
                                    kernel_size=3,
                                    padding="same",
                                    kernel_initializer=initializer)(function_2)

    X = K.layers.concatenate([X, X_convolution])
    nb_filters = nb_filters + growth_rate

    return X, nb_filters
