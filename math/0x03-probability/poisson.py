#!/usr/bin/env python3
"""Its a file tha cotain a class that represents a poisson distribution"""


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
        """Method the value of the PMF Probability Mass Functions
        for a given number of successes"""
        if k < 0:
            return 0
        k = int(k)
        PMF = ((self.e**-self.lambtha)*(self.lambtha ** k))/factorial(k)
        return PMF


def factorial(n):
    """Function to calculate the factorial from a number"""
    if n < 2:
        return 1
    return n * factorial(n-1)
