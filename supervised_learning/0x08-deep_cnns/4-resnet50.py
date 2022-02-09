#!/usr/bin/env python3
"""File that contains the function resnet50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function that that builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015):
    """
    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()
    function = "relu"

    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=7,
                             padding='same',
                             strides=2,
                             kernel_initializer=initializer)(X)

    Normalization_1 = K.layers.BatchNormalization()(conv_1)

    fuction_1 = K.layers.Activation(function)(Normalization_1)

    pool_1 = K.layers.MaxPool2D(pool_size=3,
                                strides=2,
                                padding='same')(fuction_1)

    projection_1 = projection_block(pool_1, [64, 64, 256], 1)

    identity2_2 = identity_block(projection_1, [64, 64, 256])
    indentity2_3 = identity_block(identity2_2, [64, 64, 256])

    projection_2 = projection_block(indentity2_3, [128, 128, 512])

    identity3_1 = identity_block(projection_2, [128, 128, 512])
    identity3_2 = identity_block(identity3_1, [128, 128, 512])
    identity3_3 = identity_block(identity3_2, [128, 128, 512])

    projection_3 = projection_block(identity3_3, [256, 256, 1024])

    identity4_1 = identity_block(projection_3, [256, 256, 1024])
    identity4_2 = identity_block(identity4_1, [256, 256, 1024])
    identity4_3 = identity_block(identity4_2, [256, 256, 1024])
    identity4_4 = identity_block(identity4_3, [256, 256, 1024])
    identity4_5 = identity_block(identity4_4, [256, 256, 1024])

    projection_4 = projection_block(identity4_5, [512, 512, 2048])

    identity5_1 = identity_block(projection_4, [512, 512, 2048])
    identity5_2 = identity_block(identity5_1, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         padding='same')(identity5_2)

    full_connection = K.layers.Dense(1000,
                                     activation='softmax',
                                     kernel_initializer=initializer)(avg_pool)

    model = K.models.Model(inputs=X, outputs=full_connection)

    return model
