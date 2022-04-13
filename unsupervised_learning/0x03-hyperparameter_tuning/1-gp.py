#!/usr/bin/env python3
"""File that contains the class GaussianProcess"""
import numpy as np


class GaussianProcess():
    """
    Class that represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor initializing method
        Args:
            X_init is a numpy.ndarray of shape (t, 1) representing the
            inputs already sampled with the black-box function
            Y_init is a numpy.ndarray of shape (t, 1) representing the
            outputs of the black-box function for each input in X_init
            t is the number of initial samples
            l is the length parameter for the kernel
            sigma_f is the standard deviation given to the output of the
            black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Function that calculates the covariance kernel matrix between
        two matrices
        Args:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1)
        Returns: the covariance kernel matrix as a numpy.ndarray of
        shape (m, n)
        """
        # formula κ(xi,xj)=σ^2f exp(−12l2(xi−xj)T(xi−xj))(10)
        # source: http://krasserm.github.io/2018/03/19/gaussian-processes/
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        cov_K_M = self.sigma_f ** 2 * np.exp(-0.5 / self.l**2 * sqdist)

        return cov_K_M

    def predict(self, X_s):
        """
        Public instance method that predicts the mean and standard deviation of
        points in a Gaussian process
        Args:
            X_s is a numpy.ndarray of shape (s, 1) containing all of the points
            whose mean and standard deviation should be calculated
                s is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for each
            point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the variance for
            each point in X_s, respectively
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # formula mu: μ∗ =K∗.T Ky^−1y
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)

        # formula sigma: Σ∗ =K∗∗ − K∗.T Ky^−1 K∗
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        cov_s = cov_s.diagonal()

        return mu_s, cov_s
