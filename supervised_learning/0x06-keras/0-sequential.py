#!/usr/bin/env python3
"""File that contains the function build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that builds a neural network with the Keras library
    Args:
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """
    model = K.Sequential()

    regularizer = K.regularizers.L2(lambtha)

    model.add(K.layers.Dense(layers[0],
                             input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=regularizer))

    layer_activation = zip(layers[1:], activations[1:])

    for layer, act_f in layer_activation:
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layer,
                                 activation=act_f,
                                 kernel_regularizer=regularizer))

    return model
