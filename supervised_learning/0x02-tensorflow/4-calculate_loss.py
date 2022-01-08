#!/usr/bin/env python3
"""File that contains the function calculate_loss"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Function that calculates he softmax cross-entropy loss of
    a prediction:
    Args:
    y: is a placeholder for the labels of the input data
    y_pred: is a tensor containing the network’s predictions
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    return loss
