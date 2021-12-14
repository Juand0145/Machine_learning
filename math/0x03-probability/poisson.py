#!/usr/bin/env python3
"""Its a file tha cotain a class that represents a poisson distribution"""


class Poisson():
    """Class that represents a poisson distribution"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, lambtha=1.):
        import numpy as np
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = np.mean(data)

    def pmf(self, k):
        import math
        """Method the value of the PMF Probability Mass Functions
        for a given number of successes"""
        if type(k) is not int:
            self.k = int(k)
        if k < 0:
            return 0

        PMF = ((self.e**-self.lambtha)*(self.lambtha ** k))/math.factorial(k)
        return PMF
