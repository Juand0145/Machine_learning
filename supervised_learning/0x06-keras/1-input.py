#!/usr/bin/env python3
"""File that contains the function build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that builds a neural network with the Keras library
    Args:
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the
    network
    activations is a list containing the activation functions used for
    each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """

    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)

    output = K.layers.Dense(layers[0],
                            activation=activations[0],
                            kernel_regularizer=regularizer)(inputs)

    hidden_layers = range(len(layers))[1:]

    for i in hidden_layers:
        dropout = K.layers.Dropout(1 - keep_prob)(output)
        output = K.layers.Dense(layers[i], activation=activations[i],
                                kernel_regularizer=regularizer)(dropout)

    model = K.Model(inputs, output)

    return model
