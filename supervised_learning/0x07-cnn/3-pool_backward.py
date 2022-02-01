#!/usr/bin/env python3
"""File that contains teh function pool_backward"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling
    layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
    the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (m, h_new, w_new, c_new) = dA.shape
    (kh, kw) = kernel_shape
    sh, sw = stride

    dAprev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vstart = h * sh
                    vend = vstart + kh
                    hstart = w * sw
                    hend = hstart + kw

                    if mode == 'max':
                        aslice = a_prev[vstart:vend, hstart:hend, c]
                        mask = (aslice == np.max(aslice))
                        dAprev[i, vstart:vend,
                               hstart:hend,
                               c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == 'avg':
                        da = dA[i, h, w, c]
                        shape = kernel_shape
                        average = da / (kh * kw)
                        Z = np.ones(shape) * average
                        dAprev[i,
                               vstart:vend,
                               hstart:hend, c] += Z
    return dAprev
