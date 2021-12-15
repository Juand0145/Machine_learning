#!/usr/bin/env python3
"""Fila that contain the class that represents a normal distribution"""


class Normal:
    """Class that represents a normal distribution"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data)/len(data)
            self.mean = float(mean)

            standard_diviations = []
            for x in data:
                value = pow(x - self.mean, 2)
                standard_diviations.append(value)

            stddev = pow(sum(standard_diviations)/len(data), 1/2)
            self.stddev = float(stddev)

    def z_score(self, x):
        """Instance method tha calculate the z standard scores"""
        z = (x - self.mean)/self.stddev
        return z

    def x_value(self, z):
        """Instance method that calculate x values"""
        x = (z * self.stddev) + self.mean
        return x

    def pdf(self, x):
        """Normal Distribution PDF"""
        function_part1 = 1 / (self.stddev * pow(2 * self.pi, 1/2))
        function_part2 = pow(x - self.mean, 2) / (2 * pow(self.stddev, 2))

        pdf = function_part1 * pow(self.e, -function_part2)
        return pdf

    def cdf(self, x):
        """Normal Distribution CDF"""
        function_part1 = (x - self.mean)/(self.stddev*pow(2, 1/2))

        cdf = (1/2)*(1+erf(function_part1))
        return cdf


def erf(z):
    """Function to aproximate the error function"""
    pi = 3.1415926536

    parameters = (z - (pow(z, 3)/3) + (pow(z, 5)/10) -
                  (pow(z, 7)/42) + (pow(z, 9)/216))

    erf = (2/pow(pi, 1/2)) * parameters
    return erf
