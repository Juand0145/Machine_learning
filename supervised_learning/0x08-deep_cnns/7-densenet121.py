#!/usr/bin/env python3
"""File that contain the function densenet121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Function that  builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks
    Args:
    growth_rate is the growth rate
    compression is the compression factor
    Returns: the keras model
    """
    X = K.Input((224, 224, 3))
    initializer = K.initializers.he_normal()
    function = "relu", "softmax"

    normalization_0 = K.layers.BatchNormalization()(X)
    function_0 = K.layers.Activation(function[0])(normalization_0)

    conv_1 = K.layers.Conv2D(filters=2 * growth_rate,
                             kernel_size=7,
                             strides=2,
                             padding="same",
                             kernel_initializer=initializer)(function_0)

    max_pool_1 = K.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    padding="same")(conv_1)

    dense_1, nf_1 = dense_block(max_pool_1, 2*growth_rate, growth_rate, 6)
    trans_1, nf_2 = transition_layer(dense_1, nf_1, compression)

    dense_2, nf_3 = dense_block(trans_1, nf_2, growth_rate, 12)
    trans_2, nf_4 = transition_layer(dense_2, nf_3, compression)

    dense_3, nf_5 = dense_block(trans_2, nf_4, growth_rate, 24)
    trans_3, nf_6 = transition_layer(dense_3, nf_5, compression)

    dense_4, nf_7 = dense_block(trans_3, nf_6, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         padding="same")(dense_4)

    full_connection = K.layers.Dense(1000,
                                     activation=function[1],
                                     kernel_initializer=initializer)(avg_pool)

    model = K.models.Model(inputs=X, outputs=full_connection)

    return model
