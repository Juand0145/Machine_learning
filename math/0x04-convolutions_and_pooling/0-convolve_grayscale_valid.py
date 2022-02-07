#!/usr/bin/env python3
"""File that contains the function convolve_grayscale_valid"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images:
    Args:
    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    conv_h = h - kh + 1
    conv_w = w - kw + 1

    convolved_image = np.zeros((m, conv_h, conv_w))

    image = np.arange(m)

    for i in range(conv_h):
        for j in range(conv_w):
            convolved_image[image, i, j] = np.sum(
                images[image, i:kh+i, j:kw+j] * kernel, axis=(1, 2))

    return convolved_image
