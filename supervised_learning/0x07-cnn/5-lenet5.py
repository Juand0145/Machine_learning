#!/usr/bin/env python3
"""File that contains the function lenet5"""
import tensorflow.keras as K


def lenet5(X):
    """
    FUnction that builds a modified version of the LeNet-5 architecture
    using tensorflow
    Args:
    X is a K.Input of shape (m, 28, 28, 1) containing the input
    images for the network
            m is the number of images
        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with valid padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes
        All layers requiring initialization should initialize their kernels
        with
        the he_normal initialization method
        All hidden layers requiring activation should use the relu activation
        function
        you may import tensorflow.keras as K
        Returns: a K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics
    """
    initializer = K.initializers.VarianceScaling(scale=2.0)
    act_function = "relu"

    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=5,
                         padding="same",
                         activation=act_function,
                         kernel_initializer=initializer)(X)

    F1 = K.layers.MaxPooling2D(pool_size=[2, 2],
                               strides=2)(C1)

    C2 = K.layers.Conv2D(filters=16,
                         kernel_size=5,
                         padding='valid',
                         activation=act_function,
                         kernel_initializer=initializer)(F1)

    F2 = K.layers.MaxPooling2D(pool_size=[2, 2],
                               strides=2)(C2)

    flatten = K.layers.Flatten()(F2)

    F_C1 = K.layers.Dense(units=120,
                          activation=act_function,
                          kernel_initializer=initializer)(flatten)

    F_C2 = K.layers.Dense(units=84,
                          activation=act_function,
                          kernel_initializer=initializer)(F_C1)

    F_C3 = K.layers.Dense(units=10,
                          kernel_initializer=initializer)(F_C2)

    model = K.models.Model(X, F_C3)

    adam = K.optimizers.Adam()

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
