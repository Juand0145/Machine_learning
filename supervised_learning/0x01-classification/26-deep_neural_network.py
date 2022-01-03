#!/usr/bin/env python3
"""File that contains DeepNeuralNetwork class"""
import numpy as np
import matplotlib.pyplot as plt


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
        def sigmoid(z):
            """Function that calculates the sigmoid o a entry"""
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid

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
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        """
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
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
        A = self.forward_prop(X)

        dicision_boundary = np.where(A[0] < 0.5, 0, 1)
        cost = self.cost(Y, A[0])

        return dicision_boundary, cost

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
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise TypeError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 and step > iterations:
                raise("step must be positive and <= iterations")

        cost_points = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(Y, cache, alpha)

            if i == 0 and verbose:
                print("Cost after 0 iterations: {}".format(cost))
            elif i % step == 0 and verbose:
                print("Cost after {} iterations: {}".format(i, cost))
            elif i + 1 == iterations:
                print("Cost after {} iterations: {}".format(iterations, cost))

            if graph:
                cost_points.append(cost)

        if graph:
            plt.plot(cost_points)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

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

        new_object = DeepNeuralNetwork(atributes[0], atributes[1])
        new_object.__L = atributes[2]
        new_object.__cache = atributes[3]
        new_object.__weights = atributes[4]

        return new_object
