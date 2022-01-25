#!/usr/bin/env python3
"""File that contains the functions save_config and
load_config"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Function that saves a model’s configuration in JSON format
    Args:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration
    should be saved to
    Returns: None
    """
    with open(filename, "w") as file:
        file.write(network.to_json())


def load_config(filename):
    """
    Function that loads a model with a specific configuration
    Args:
    filename is the path of the file containing the model’s
    configuration in JSON format
    Returns: the loaded model
    """
    with open(filename, "r") as file:
        network_configuration = file.read()

    model = K.models.model_from_json(network_configuration)

    return model
