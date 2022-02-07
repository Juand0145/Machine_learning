#!/usr/bin/env python3
"""File that contains the function convolve_channels"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that  that performs pooling on images
    Args:
    images is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel
    shape for the pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    conv_h = int(1 + ((h - kh) / sh))
    conv_w = int(1 + ((w - kw) / sw))

    convolved_image = np.zeros((m, conv_h, conv_w, c))

    image = np.arange(m)

    for i in range(conv_h):
        for j in range(conv_w):
            if mode == 'max':
                convolved_image[image, i, j] = (np.max(
                    images[image,
                           i * sh:((i * sh) + kh),
                           j * sw:((j * sw) + kw)],
                    axis=(1, 2)))

            if mode == 'avg':
                convolved_image[image, i, j] = (np.mean(
                    images[image,
                           i * sh:((i * sh) + kh),
                           j * sw:((j * sw) + kw)],
                    axis=(1, 2)))

    return convolved_image
