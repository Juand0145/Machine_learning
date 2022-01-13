#!/usr/bin/env python3
"""File that contains the function update_variables_RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Function that updates a variable using the RMSProp
    optimization algorithm
    Args:
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var
    Returns: the updated variable and the new moment, respectively
    """
    Sdv = (beta2 * s) + (1 - beta2) * grad ** 2

    var = var - alpha * grad / (pow(Sdv, 0.5) + epsilon)

    return var, Sdv
