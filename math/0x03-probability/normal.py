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
