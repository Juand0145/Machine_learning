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
        self.nx = nx

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

    def cost(self, Y, A):
        """
        Public method that calculates the cost using a logistic regresión
        Args:
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        """
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """
        Public method that hat defines a single neuron performing binary
        classification
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            - nx is the number of input features to the neuron
            - m is the number of examples
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        """
        forward_propagation = self.forward_prop(X)
        cost_function = self.cost(Y, self.__A)
        decision_boundary = np.where(self.__A < 0.5, 0, 1)

        return decision_boundary, cost_function

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Public method that calculates one pass of gradient descent on the
        neuron
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        alpha: is the learning rate
        """
        m = Y.shape[1]
        dCodz = A - Y
        dw = (1 / m) * np.matmul(X, dCodz.T)
        db = (1 / m) * np.sum(dCodz)

        self.__W = self.__W - (dw * alpha).T
        self.__b = self.__b - db * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Public method that allows us to train our neuron and find the
        respective weight nad bias
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data        
        """

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)


def sigmoid(z):
    """
    Function that calculate the sigmoid function:
    Args:
    z: value to apply the sigmoid function
    """
    sig = 1 / (1 + np.exp(-z))
    return sig
