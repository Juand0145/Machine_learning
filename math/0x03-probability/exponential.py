#!/usr/bin/env python3
"""File that contain the class that represents an
exponential distribution"""


class Exponential:
    """Class that represents an exponential distribution"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """Method that value the pdf of exponential distribution"""
        if x < 0:
            return 0

        pdf = self.lambtha * pow(self.e, -1 * self.lambtha * x)
        return pdf
