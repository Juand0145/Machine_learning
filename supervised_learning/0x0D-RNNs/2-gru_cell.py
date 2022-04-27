#!/usr/bin/env python3
"""File that contains the Class GRUCell """
import numpy as np


class GRUCell():
    """Class that represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        class constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh,
        by that represent the weights and biases of the cell
          Wz and bz are for the update gate
          Wr and br are for the reset gate
          Wh and bh are for the intermediate hidden state
          Wy and by are for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation for one
        time step
         Args:
         x_t is a numpy.ndarray of shape (m, i) that contains the data
         input for the cell
            m is the batche size for the data
          h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state
          The output of the cell should use a softmax activation function
          Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        def softmax(x):
            """Function that perform the softmax function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        def sigmoid(x):
            """Function that perform the sigmoid function"""
            return 1 / (1 + np.exp(-x))

        # previous hidden cell state
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        # update gate z vector (TYPE 1)
        zt = sigmoid((h_x.T @ self.Wz) + self.bz)

        # reset gate vector (TYPE 1)
        rt = sigmoid((h_x.T @ self.Wr) + self.br)

        # cell operation after updated z
        h_x = np.concatenate(((rt * h_prev).T, x_t.T), axis=0)

        # x_t activated via tanh to get candidate activation vector
        ht_c = np.tanh((h_x.T @ self.Wh) + self.bh)

        # compute output vector
        h_next = (1 - zt) * h_prev + zt * ht_c

        # final output of the cell
        y = softmax((h_next @ self.Wy) + self.by)

        return h_next, y
