#!/usr/bin/env python3
"""File that contains the function dropout_gradient_descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that updates the weights of a neural network with Dropout
    regularization using gradient descent
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of each layer of
    the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    All layers use thetanh activation function except the last, which uses the
    softmax activation function
    The weights of the network should be updated in place
    """
    m = Y.shape[1]
    Weight_copy = weights.copy()
    Layers = range(L + 1)[1:L + 1]

    for i in reversed(Layers):
        A = cache["A" + str(i)]

        if i == L:
            dZ = A - Y
            dW = (np.matmul(cache["A" + str(i - 1)], dZ.T) / m).T

        else:
            dW2 = np.matmul(Weight_copy["W" + str(i + 1)].T, dZ2)
            dtanh = 1 - (pow(A, 2))

            dZ = dW2 * dtanh
            dZ = dZ * cache["D" + str(i)]
            dZ = dZ/keep_prob

            dW = np.matmul(dZ, cache["A" + str(i - 1)].T) / m

        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W" + str(i)] = (Weight_copy["W" + str(i)] - (alpha * dW))
        weights["b" + str(i)] = Weight_copy["b" + str(i)] - (alpha * db)

        dZ2 = dZ
