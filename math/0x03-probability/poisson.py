#!/usr/bin/env python3
"""Its a file tha cotain a class that represents a poisson distribution"""


from math import factorial


class Poisson:
    """Class that represents a poisson distribution"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """PMF Probability Mass Functions"""
        if type(k) is not int:
            self.k = int(k)
        if k < 0:
            return 0

        pmf = ((self.e**-self.lambtha)*(self.lambtha ** k))/factorial(k)
        return pmf
