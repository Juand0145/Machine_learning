#!/usr/bin/env python3
"""File that contains the clas NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """Class that defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """
        class constructor
        Args:
            nx is the number of input features
            nodes is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nx < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.randn(nodes).reshape(1, nodes)
        self.b2 = 0
        self.A2 = 0
