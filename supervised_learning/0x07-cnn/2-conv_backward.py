#!/usr/bin/env python3
"""File that contains the function conv_backward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Function that performs back propagation over a convolutional
    layer of a neural network
    Args:
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the unactivated output of
    the convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh is the filter height
        kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding is a string that is either same or valid, indicating the type of
    padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    """
    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride

    ph, pw = 0, 0
    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_prev_padiing = np.pad(A_prev, pad_width=(
        (0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant")
    dA_prev_padiing = np.pad(dA_prev, pad_width=(
        (0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    for i in range(m):
        a_prev_pading = A_prev_padiing[i]
        da_prev_pading = dA_prev_padiing[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vstart = h * sh
                    vend = vstart + kh
                    hstart = w * sw
                    hend = hstart + kw
                    aslice = a_prev_pading[vstart:vend, hstart:hend]
                    da_prev_pading[vstart:vend,
                                   hstart:hend] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += aslice * dZ[i, h, w, c]

        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pading[ph:-ph, pw:-pw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pading

    return dA_prev, dW, db
