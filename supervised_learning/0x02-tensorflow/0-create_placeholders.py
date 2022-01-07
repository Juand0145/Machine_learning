#!/usr/bin/env python3
"""File that contains the class create_placeholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders, x and y, for
    the neural network:
    Args:
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier
    Returns: placeholders named x and y, respectively
        x is the placeholder for the input data to the neural network
        y is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y