#!/usr/bin/env python3
"""File that contains the function calculate_accuracy"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction
    Args:
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    max_prediction_index = tf.argmax(y_pred, 1)
    equal = tf.equal(tf.argmax(y, 1), max_prediction_index)

    accuaracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return accuaracy
