#!/usr/bin/env python3
"""File that contain the function sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    Function that calculates the sensitivity for each class
    in a confusion matrix
    Args:
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the
    sensitivity of each class
    """

    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    recall = TP/(TP + FN)

    return recall
