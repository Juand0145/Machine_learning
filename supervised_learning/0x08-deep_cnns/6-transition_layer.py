#!/usr/bin/env python3
"""File that contain the function transition_layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer as described in
    Densely Connected Convolutional Networks
    Args:
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Returns: The output of the transition layer and the number of
    filters within the output, respectively
    """
    initializer = K.initializers.he_normal()
    function = "relu"
    number_filters = int(nb_filters * compression)

    normalization_1 = K.layers.BatchNormalization()(X)

    function_1 = K.layers.Activation(function)(normalization_1)

    convolution = K.layers.Conv2D(filters=number_filters,
                                  kernel_size=1,
                                  padding="same",
                                  kernel_initializer=initializer)(function_1)

    average_pool = K.layers.AveragePooling2D(pool_size=2,
                                             strides=2,
                                             padding="same")(convolution)

    return average_pool, number_filters
