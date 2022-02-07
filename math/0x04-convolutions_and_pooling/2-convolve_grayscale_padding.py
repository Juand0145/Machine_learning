#!/usr/bin/env python3
"""File that contains the function convolve_grayscale_same"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a valid convolution on grayscale images:
    Args:
    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    pw = padding[1]
    ph = padding[0]

    conv_h = h + (2 * ph) - kh + 1
    conv_w = w + (2 * pw) - kw + 1

    padding_image = np.pad(images, pad_width=(
        (0, 0), (ph, ph), (pw, pw)), mode='constant')

    convolved_image = np.zeros((m, conv_h, conv_w))

    image = np.arange(m)

    for i in range(conv_h):
        for j in range(conv_w):
            convolved_image[image, i, j] = (
                np.sum(padding_image[image, i:kh+i, j:kw+j] * kernel,
                       axis=(1, 2)))

    return convolved_image
