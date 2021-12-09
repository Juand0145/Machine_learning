#!/usr/bin/env python3
"""Is a function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if type(poly) is not list or type(C) not in (int, float):
        return None

    integrated_coefficient = [C]

    for power, coefficient in enumerate(poly):
        if coefficient == 0:
            integrated_coefficient.append(0)

        elif power != 0:
            new_coefficient = coefficient / (power + 1)
            integrated_coefficient.append(new_coefficient)

        else:
            new_coefficient = coefficient
            integrated_coefficient.append(new_coefficient)

    return integrated_coefficient
