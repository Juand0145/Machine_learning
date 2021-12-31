#!/usr/bin/env python3
"""File that contains DeepNeuralNetwork class"""
import numpy as np


class DeepNeuralNetwork:
    """Class that that defines a deep neural network performing binary
    classification:"""

    def __init__(self, nx, layers):
        """
        class constructor
        Arg:
        nx: is the number of input features
        layers: is a list representing the number of nodes in each layer of the
        network
        """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        if any(list(map(lambda x: x <= 0, layers))):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            key = "W{}".format(i + 1)
            if i == 0:
                self.__weights[key] = np.random.randn(layers[i],
                                                      nx)*np.sqrt(2/nx)
                self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.__weights[key] = (np.random.randn(layers[i],
                                                       layers[i - 1]) *
                                       np.sqrt(2/layers[i - 1]))
                self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """The number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """A dictionary to hold all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """ A dictionary to hold all weights and biased of the network"""
        return self.__weights
