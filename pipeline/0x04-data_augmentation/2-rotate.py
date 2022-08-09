#!/usr/bin/env python3
"""File that contains the function rotate_image"""
import tensorflow as tf

def rotate_image(image):
    """
    Function that rotates an image by 90 degrees counter-clockwise
    Args:
      image is a 3D tf.Tensor containing the image to rotate
    Returns the rotated image
    """
    return tf.image.rot90(image)
