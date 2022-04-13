#!/usr/bin/env python3
"""File that contains the class BayesianOptimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that  performs Bayesian optimization on a noiseless
    1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor initializer
            Args:
                f is the black-box function to be optimized
                X_init is a numpy.ndarray of shape (t, 1) representing the
                inputs already sampled with the black-box function
                Y_init is a numpy.ndarray of shape (t, 1) representing the
                outputs of the black-box function for each input in X_init
                t is the number of initial samples
                bounds is a tuple of (min, max) representing the bounds of the
                space in which to look for the optimal point
                ac_samples is the number of samples that should be analyzed
                during acquisition
                l is the length parameter for the kernel
                sigma_f is the standard deviation given to the output of the
                black-box function
                xsi is the exploration-exploitation factor for acquisition
                minimize is a bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f

        self.gp = GP(X_init, Y_init, l, sigma_f)

        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        self.xsi = xsi

        self.minimize = minimize

    def acquisition(self):
        """
        Public instance method that calculates the next best sample location
        Args:
            Uses the Expected Improvement acquisition function
        Returns: X_next, EI
            X_next is a numpy.ndarray of shape (1,) representing the next
            best sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing the
            expected improvement of each potential sample
        """
        from scipy.stats import norm
        # source: http://krasserm.github.io/2018/03/21/bayesian-optimization/
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu - Y_sample - self.xsi

        Z = np.zeros(sigma.shape[0])
        for i in range(sigma.shape[0]):
            # formula if σ(x)>0 : μ(x)−f(x+)−ξ / σ(x)
            if sigma[i] > 0:
                Z[i] = imp[i] / sigma[i]
            # formula if σ(x)=0
            else:
                Z[i] = 0
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Public instance method  that optimizes the black-box function
        Args:
            iterations is the maximum number of iterations to perform
        Returns: X_opt, Y_opt
            X_opt is a numpy.ndarray of shape (1,) representing the optimal
            point
            Y_opt is a numpy.ndarray of shape (1,) representing the optimal
            function value
        """
        X_all_s = []
        for i in range(iterations):
            x_opt, _ = self.acquisition()
            if x_opt in X_all_s:
                break

            y_opt = self.f(x_opt)

            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt
