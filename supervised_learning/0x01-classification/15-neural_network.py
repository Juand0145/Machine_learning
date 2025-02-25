#!/usr/bin/env python3
"""File that contains the clas NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt


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
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """
        Public method that calculates the forward propagation
        of the neural network
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Returns the private attributes __A1 and __A2, respectively
        """
        def sigmoid(z):
            """
            Function that calculates the sigmoid function
            Args:
            z: number to asign the sigmoid
            """
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid

        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(z1)

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(z2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Public method that calculates the cost of the model using
        logistic regression
        Args:
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        """
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """
        Public method that evaluates the neural network’s predictions
        Args:
        X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        """
        forward_propagation = self.forward_prop(X)
        dicision_boundary = np.where(self.__A2 < 0.5, 0, 1)

        cost_function = self.cost(Y, self.__A2)

        return dicision_boundary, cost_function

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Public method that calculates one pass of gradient descent
        on the neural network
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = (1/m) * np.matmul(A1, dz2.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1/m) * np.matmul(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - (dw2 * alpha).T
        self.__b2 = self.__b2 - db2 * alpha

        self.__W1 = self.__W1 - dw1 * alpha
        self.__b1 = self.__b1 - db1 * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Public method that trains the neural network
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations: is the number of iterations to train over
        alpha: is the learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 and step > iterations:
                raise("step must be positive and <= iterations")

        cost_points = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

            if i == 0 and verbose:
                print(f"Cost after 0 iterations: {cost}")
            elif i % step == 0 and verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if graph:
                cost_points.append(cost)

        if graph:
            plt.plot(cost_points)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)
