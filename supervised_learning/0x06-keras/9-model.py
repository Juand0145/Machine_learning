#!/usr/bin/env python3
"""File that contains the functions save_model and
load_model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Function that saves an entire model
    Args:
    network is the model to save
    filename is the path of the file that the model should be saved to
    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Function that load an entire model
    Args:
    filename is the path of the file that the model should be loaded from
    Returns: the loaded model
    """
    load = K.models.load_model(filename)
    return load
