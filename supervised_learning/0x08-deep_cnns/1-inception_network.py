#!/usr/bin/env python3
"""File that contain the function inception_network"""
import tensorflow.keras as K


def inception_network():
    """
    Function that builds the inception network as described
    in Going Deeper with Convolutions (2014)
    Returns: the keras model
    """
    inception_block = __import__('0-inception_block').inception_block

    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()
    function = "relu"

    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=7,
                             strides=(2, 2),
                             padding="same",
                             activation=function,
                             kernel_initializer=initializer)(X)

    max_pool_1 = K.layers.MaxPool2D(pool_size=[3, 3],
                                    strides=(2, 2),
                                    padding="same")(conv_1)

    conv_2P = K.layers.Conv2D(filters=64,
                              kernel_size=1,
                              padding="valid",
                              activation=function,
                              kernel_initializer=initializer)(max_pool_1)

    conv_2 = K.layers.Conv2D(filters=192,
                             kernel_size=3,
                             padding="same",
                             activation=function,
                             kernel_initializer=initializer)(conv_2P)

    max_pool_2 = K.layers.MaxPool2D(pool_size=[3, 3],
                                    strides=(2, 2),
                                    padding="same",)(conv_2)

    inception_3a = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    max_pool_3 = K.layers.MaxPooling2D(pool_size=[3, 3],
                                       strides=(2, 2),
                                       padding="same")(inception_3b)

    inception_4a = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    max_pool_4 = K.layers.MaxPool2D(pool_size=[3, 3],
                                    strides=(2, 2),
                                    padding="same")(inception_4e)

    inception_5a = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    average_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                             strides=(1, 1),
                                             padding='valid')(inception_5b)

    dropout = K.layers.Dropout(0.4)(average_pool)

    full_connection = K.layers.Dense(1000,
                                     activation="softmax",
                                     kernel_initializer=initializer)(dropout)

    model = K.models.Model(inputs=X,
                           outputs=full_connection)

    return model
