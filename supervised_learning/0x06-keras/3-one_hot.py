#!/usr/bin/env python3
"""File that contains the function one_hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function that converts a label vector into a one-hot matrix
    Args:
    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot
