#!/usr/bin/env python3
"""File That contains the function l2_reg_gradient_descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that calculates  updates the weights and biases of a neural
    network using gradient descent with L2 regularization:
    Args:
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    """
    m = Y.shape[1]
    W_copy = weights.copy()

    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]

        if i == L - 1:
            dZ = cache[f"A{i + 1}"] - Y
            dW = (np.matmul(cache[f"A{i}"], dZ.T) / m).T

            dW_L2 = dW + (lambtha / m) * W_copy[f"W{i + 1}"]
            db = np.sum(dZ, axis=1, keepdims=True) / m

        else:
            dW2 = np.matmul(W_copy[f"W{i + 2}"].T, dZ2)
            tanh = 1 - (A * A)
            dZ = dW2 * tanh
            dW = np.matmul(dZ, cache[f"A{i}"].T) / m

            dW_L2 = dW + (lambtha / m) * W_copy[f"W{i + 1}"]
            db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[f"W{i + 1}"] = (W_copy[f"W{i + 1}"]) - (alpha * dW_L2)
        weights[f"b{i + 1}"] = W_copy[f"b{i + 1}"] - (alpha * db)
        dZ2 = dZ
