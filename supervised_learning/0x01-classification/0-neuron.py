#!/usr/bin/env python3
"""File that contains the class Neuron"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Neuron clas constructor
        args:
            nx: is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError ("nx must be an integer")
        if nx < 1:
            raise ValueError ("nx must be a positive integer")

        W = np.random.randn(nx)
        self.W = W.reshape(1, nx)
        self.b = 0
        self.A = 0
