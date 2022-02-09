#!/usr/bin/env python3
"""File that contains the function projection_block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Function that that builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015)
    Args:
        A_prev: the output from the previous layer
        filters: tuple or list containing the following filters
                 F11: the number of filters in the 1st 1x1 convolution
                 F3: the number of filters in the 3x3 convolution
                 F12: the number of filters in the 2nd 1x1 convolution
        s: stride of the first convolution in both the main path and
        the shortcut connection
    """
    initializer = K.initializers.he_normal()
    function = 'relu'
    F11, F3, F12 = filters

    conv_1 = K.layers.Conv2D(filters=F11,
                             kernel_size=1,
                             strides=s,
                             padding='same',
                             kernel_initializer=initializer)(A_prev)

    normalize_1 = K.layers.BatchNormalization(axis=3)(conv_1)

    function_1 = K.layers.Activation(function)(normalize_1)

    conv_2 = K.layers.Conv2D(filters=F3,
                             kernel_size=3,
                             padding='same',
                             kernel_initializer=initializer)(function_1)

    normalize_2 = K.layers.BatchNormalization(axis=3)(conv_2)

    function_2 = K.layers.Activation(function)(normalize_2)

    conv_3 = K.layers.Conv2D(filters=F12,
                             kernel_size=1,
                             padding='same',
                             kernel_initializer=initializer)(function_2)

    conv1_proj = K.layers.Conv2D(filters=F12,
                                 kernel_size=1,
                                 strides=s,
                                 padding='same',
                                 kernel_initializer=initializer)(A_prev)

    normalization_3 = K.layers.BatchNormalization(axis=3)(conv_3)

    normaliztion_4 = K.layers.BatchNormalization(axis=3)(conv1_proj)
    add = K.layers.Add()([normalization_3, normaliztion_4])

    final_relu = K.layers.Activation(function)(add)

    return final_relu
