#!/usr/bin/env python3
"""File that contains the function pool_forward"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs forward propagation over a pooling layer
    of a neural network
    Args:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the
    pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating
    hether to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    conv_height = int((h_prev - kh) / sh) + 1
    conv_wide = int((w_prev - kw) / sw) + 1

    convolutionary_image = np.zeros((m, conv_height, conv_wide, c_prev))

    for i in range(conv_height):
        for j in range(conv_wide):
            if mode == 'max':
                convolutionary_image[:, i, j] = (np.max(A_prev[:,
                                                               i * sh:((i * sh) + kh),
                                                               j * sw:((j * sw) + kw)],
                                                        axis=(1, 2)))
            elif mode == 'avg':
                convolutionary_image[:, i, j] = (np.mean(A_prev[:,
                                                                i * sh:((i * sh) + kh),
                                                                j * sw:((j * sw) + kw)],
                                                         axis=(1, 2)))

    return convolutionary_image
