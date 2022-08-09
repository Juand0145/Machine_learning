#!/usr/bin/env python3
"""File that contains the function shear_image"""
import tensorflow as tf

def shear_image(image, intensity):
    """
    Function that randomly shears an image
    Args:
      image is a 3D tf.Tensor containing the image to shear
      intensity is the intensity with which the image should be sheared
    Returns the sheared image
    """
    return tf.keras.preprocessing.image.random_shear(
        x = image,
        intensity = intensity
    )
