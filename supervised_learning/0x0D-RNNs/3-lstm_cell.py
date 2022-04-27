#!/usr/bin/env python3
"""File that contains the class LSTMCell"""
import numpy as np


class LSTMCell():
    """Class that represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        class constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc,
        bo, by that represent the weights and biases of the cell
          Wf and bf are for the forget gate
          Wu and bu are for the update gate
          Wc and bc are for the intermediate cell state
          Wo and bo are for the output gate
          Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Public instance Method taht performs forward propagation for one
        time step
        Args:
          x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
            m is the batche size for the data
          h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state
          c_prev is a numpy.ndarray of shape (m, h) containing the previous
          cell state
        Returns: h_next, c_next, y
          h_next is the next hidden state
          c_next is the next cell state
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

        # forget gate activation vector
        ft = sigmoid((h_x.T @ self.Wf) + self.bf)

        # input/update gate activation vector
        it = sigmoid((h_x.T @ self.Wu) + self.bu)

        # candidate value
        cct = np.tanh((h_x.T @ self.Wc) + self.bc)
        c_next = ft * c_prev + it * cct

        # output gate
        ot = sigmoid((h_x.T @ self.Wo) + self.bo)

        # compute hidden state
        h_next = ot * np.tanh(c_next)

        # final output of the cell
        y = softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
