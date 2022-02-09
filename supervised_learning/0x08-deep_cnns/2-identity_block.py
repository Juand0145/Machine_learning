#!/usr/bin/env python3
"""File that contains the function identity_block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Function that builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)
    Args:
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    """
    initializer = K.initializers.he_normal()
    function = "relu"
    F11, F3, F12 = filters

    conv_1 = K.layers.Conv2D(filters=F11,
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=initializer)(A_prev)

    normalization_1 = K.layers.BatchNormalization(axis=3)(conv_1)

    function_1 = K.layers.Activation(function)(normalization_1)

    conv_2 = K.layers.Conv2D(filters=F3,
                             kernel_size=3,
                             padding="same",
                             kernel_initializer=initializer)(function_1)

    normalization_2 = K.layers.BatchNormalization(axis=3)(conv_2)

    function_2 = K.layers.Activation(function)(normalization_2)

    conv_3 = K.layers.Conv2D(filters=F12,
                             kernel_size=1,
                             padding="same",
                             kernel_initializer=initializer)(function_2)

    normalization_3 = K.layers.BatchNormalization(axis=3)(conv_3)

    # Add shortcut to main path, pass through a relu activation
    add = K.layers.Add()([normalization_3, A_prev])

    function_3 = K.layers.Activation(function)(add)

    return function_3
