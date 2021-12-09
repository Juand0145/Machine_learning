#!/usr/bin/env python3
"""Is a funtion that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Funtion that calculates the derivative of a polynomial"""
    if type(poly) is not list or len(poly) < 1:
        return None
    if len(poly) == 1:
        return [0]

    derivated_coefficients = []

    for power, coefficient in enumerate(poly):
        if power == 0:
            pass

        else:
            new_coefficient = coefficient * power
            derivated_coefficients.append(new_coefficient)

    return(derivated_coefficients)
