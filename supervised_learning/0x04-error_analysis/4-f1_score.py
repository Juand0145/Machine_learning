#!/usr/bin/env python3
"""File that contain the function f1_score"""
import numpy as np


def f1_score(confusion):
    """
    Function that hat calculates the precision for each
    class in a confusion matrix
    Args:
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the
    precision of each class
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    pre = precision(confusion)
    senci = sensitivity(confusion)
    f1 = 2 * (pre * senci) / (pre + senci)

    return f1
