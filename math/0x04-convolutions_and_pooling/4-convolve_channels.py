#!/usr/bin/env python3
"""File that contains the function convolve_channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
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
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    sh = stride[0]
    sw = stride[1]

    ph, pw = 0, 0

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)

    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]

    conv_h = int(((h + 2 * ph - kh) / sh) + 1)
    conv_w = int(((w + 2 * pw - kh) / sw) + 1)

    padding_image = np.pad(images, pad_width=(
        (0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    convolved_image = np.zeros((m, conv_h, conv_w))

    image = np.arange(m)

    for i in range(conv_h):
        for j in range(conv_w):
            convolved_image[image, i, j] = (np.sum(padding_image[image,
                                            i * sh:((i * sh) + kh),
                                            j * sw:((j * sw) + kw)] * kernel,
                axis=(1, 2, 3)))

    return convolved_image
