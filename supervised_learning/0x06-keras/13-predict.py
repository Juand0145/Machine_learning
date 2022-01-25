#!/usr/bin/env python3
"""File that contains the function predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function that makes a prediction using a neural network
    Args:
    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be printed
    during the prediction process
    Returns: the prediction for the data
    """
    if verbose:
        prediction = network.predict(data, verbose=1)

    else:
        prediction = network.predict(data, verbose=0)

    return prediction
