#!/usr/bin/env python3
"""Is a function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    # if type(poly) is not list or len(poly) < 1:
    #     return None

    # integrated_coefficient = [C]

    # for power, coefficient in enumerate(poly):
    #     if power != 0:
    #         new_coefficient = coefficient / (power + 1)
    #         integrated_coefficient.append(new_coefficient)

    #     else:
    #         new_coefficient = coefficient
    #         integrated_coefficient.append(new_coefficient)

    # return integrated_coefficient

    if not all(type(C) in (float, int) for c in poly) or type(C) != int:
        return None
    integral = [c/i if c % i != 0 else c//i for i, c in enumerate(poly, 1)]
    while len(integral) > 0 and integral[-1] == 0:
        integral.pop()
    return [C] + integral
