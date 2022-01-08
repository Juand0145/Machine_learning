#!/usr/bin/env python3
"""Fila that contain the function forward_prop"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for
    the neural network:
    Args:
    x: is the placeholder for the input data
    layer_sizes: is a list containing the number of nodes in
    each layer of the network
    activations: is a list containing the activation functions
    for each layer of the network
    Returns: the prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer

    layer = create_layer(x, layer_sizes[0], activations[0])

    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer
