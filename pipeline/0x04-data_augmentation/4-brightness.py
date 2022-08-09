#!/usr/bin/env python3
"""File that contains the function change_brightness"""
import tensorflow as tf

def change_brightness(image, max_delta):
    """
    Function that randomly changes the brightness of an image
    Args:
      image is a 3D tf.Tensor containing the image to change
      max_delta is the maximum amount the image should be brightened
      (or darkened)
    Returns the altered image
    """
    return tf.image.adjust_brightness(image, max_delta)
