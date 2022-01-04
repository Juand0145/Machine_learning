#!/usr/bin/env python3
"""File that contains DeepNeuralNetwork class"""
from PIL.Image import new
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.records import array


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

    def forward_prop(self, X):
        """
        Public method that calculates the forward propagation of the neural
        network
        Args:
        X is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        """
        self.__cache["A0"] = X

        for lay in range(self.__L):
            W = self.__weights
            cache = self.__cache
            X_W = np.matmul(W["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = X_W + W["b" + str(lay + 1)]
            cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))

        network_output = cache["A" + str(self.__L)]

        return network_output, cache

    def cost(self, Y, A):
        """
        Public method that Calculates the cost of the model using logistic
        regression
        Y: is a one-hot numpy.ndarray of shape (classes, m)
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        """
        # print("---------------")
        # print(f"forma de A: {A.shape}")
        # print(f"forma de Y: {Y.shape}")
        # print("---------------")

        cost = (pow(Y - A, 2)).mean()

        return cost

    def evaluate(self, X, Y):
        """
        Public method that Evaluates the neural network’s predictions
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        """
        A, cost = self.forward_prop(X)

        # print("---------------")
        # print(f"forma de A: {A.shape}")
        # print(f"Valor de A: {A[0]}")

        C = []
        for i in A.T:
            max = 0
            position = 0
            for index, value in enumerate(i):
                if value > max:
                    max = value
                    position = index
            C.append(position) 
        
        # print(f"valor de C: {C}")

        cost = self.cost(Y, A[0])
        return C, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        public method that Calculates one pass of gradient descent on
        the neural network
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        cache is a dictionary containing all the intermediary values
        of the network
        alpha is the learning rate
        """
        m = Y.shape[1]
        weights = self.weights.copy()

        for i in reversed(range(1, self.L + 1)):
            A = cache["A{}".format(i)]

            if i == self.L:
                dZ = A - Y
            else:
                dZ = np.matmul(weights["W" + str(i+1)].T, dZ) * A * (1 - A)

            dW = (1 / m) * np.matmul(dZ, cache["A" + str(i - 1)].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            W = self.weights["W" + str(i)] - (alpha * dW)
            b = self.weights["b" + str(i)] - (alpha * db)
            self.weights["W" + str(i)] = W
            self.weights["b" + str(i)] = b

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Public method that trains the deep neural network
        Args:
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations: is the number of iterations to train over
        alpha is the learning rate
        """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        else:
            if iterations < 0:
                raise ValueError('iterations must be a positive integer')

        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        else:
            if alpha < 0:
                raise ValueError('alpha must be positive')

        if graph is True or verbose is True:
            if type(step) != int:
                raise TypeError('step must be an integer')
            else:
                if step < 0 or step > iterations:
                    raise ValueError('step must be positive and <= iterations')

        cost_list = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost_list.append(self.cost(Y, A))
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))
        A, cost = self.evaluate(X, Y)
        if verbose:
            print("Cost after {} iterations: {}".format(i + 1, cost))

        if graph:
            plt.plot(list(range(iterations)), cost_list)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        Args:
            filename is the file to which the object should be saved
        """
        import pickle

        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"

        file = open(filename, "wb")

        nx = 10
        layers = [3, 1]
        L = self.__L
        cache = self.__cache
        weight = self.__weights

        object = [nx, layers, L, cache, weight]

        pickle.dump(object, file)

        file.close()

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        Args:
        filename: is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist
        """
        import pickle

        try:
            file = open(filename, "rb")
        except Exception:
            return None

        atributes = pickle.load(file)

        new_object = atributes

        return new_object
