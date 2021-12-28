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
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        W = np.random.randn(nx)
        self.__W = W.reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Returns the weights"""
        return self.__W

    @property
    def b(self):
        """Returns the bias initialized in 0"""
        return self.__b

    @property
    def A(self):
        """Returns the predictions initialized in 0"""
        return self.__A

    def forward_prop(self, X):
        """
        Public method that Calculates the forward propagation of the neuron
        Args:
        X: Is a numpy.ndarray with shape (nx, m) that contains the input data
        """
        neuron_output = np.matmul(self.__W, X) + self.__b
        ativation_function_output = sigmoid(neuron_output)

        self.__A = ativation_function_output

        return self.__A


def sigmoid(z):
    """
    Function that calculate the sigmoid function:
    Args:
    z: value to apply the sigmoid function
    """
    sig = 1 / (1 + np.exp(-z))
    return sig
