#!/usr/bin/env python3
"""File that contains the function """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    FUnction that builds a modified version of the LeNet-5 architecture
    using tensorflow
    Args:
    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot
    labels for the network
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
    tf.keras.initializers.VarianceScaling(scale=2.0)
    All hidden layers requiring activation should use the relu activation
    function
    you may import tensorflow.compat.v1 as tf
    you may NOT use tf.keras only for the he_normal method.
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
        hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    act_function = tf.nn.relu

    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=5,
                          padding="same",
                          activation=act_function,
                          kernel_initializer=initializer)(x)

    F1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                strides=2)(C1)

    C2 = tf.layers.Conv2D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation=act_function,
                          kernel_initializer=initializer)(F1)

    F2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                strides=2)(C2)

    flatten = tf.layers.Flatten()(F2)

    F_C1 = tf.layers.Dense(units=120,
                           activation=act_function,
                           kernel_initializer=initializer)(flatten)

    F_C2 = tf.layers.Dense(units=84,
                           activation=act_function,
                           kernel_initializer=initializer)(F_C1)

    F_C3 = tf.layers.Dense(units=10,
                           kernel_initializer=initializer)(F_C2)

    prediction = F_C3
    loss = tf.losses.softmax_cross_entropy(y, F_C3)

    train = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    prediction = tf.nn.softmax(prediction)

    return prediction, train, loss, accuracy
