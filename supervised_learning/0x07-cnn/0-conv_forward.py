#!/usr/bin/env python3
"""File that contains the function """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional layer of
    a neural network
    Args:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of
    padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride

    pw, ph = 0, 0

    if padding == "same":
        ph = int(((h_prev + 2 * ph - kh) / sh) + 1)
        pw = int(((w_prev + 2 * pw - kw) / sw) + 1)

    padding_image = np.pad(A_prev,
                           pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode="constant")

    c_height = int(((h_prev + 2 * ph - kh) / sh) + 1)
    c_wide = int(((w_prev + 2 * pw - kw) / sw) + 1)

    result = np.zeros((m, c_height, c_wide, c_new))

    for i in range(c_height):
        for j in range(c_wide):
            for z in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                img_slice = padding_image[:, v_start:v_end, h_start:h_end]
                kernel = W[:, :, :, z]
                result[:, i, j, z] = (np.sum(np.multiply(img_slice,
                                                         kernel),
                                             axis=(1, 2, 3)))

    Z = result + b
    return activation(Z)
