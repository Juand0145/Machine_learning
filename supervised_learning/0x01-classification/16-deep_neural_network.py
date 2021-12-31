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
        layers: is a list representing the number of nodes in each layer of the network
        """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        else:
            if any(list(map(lambda x: x <= 0, layers))):
                raise TypeError('layers must be a list of positive integers')
            if len(layers) < 1:
                raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            key = f"W{i + 1}"
            if i == 0:
                self.weights[key] = np.random.randn(layers[i],
                                                    nx)*np.sqrt(2/nx)
                self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.weights[key] = (np.random.randn(layers[i],
                                                     layers[i - 1]) *
                                     np.sqrt(2/layers[i - 1]))
                self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
