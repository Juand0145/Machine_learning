#!/usr/bin/env python3
"""File that contains the function autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional autoencoder
    Args:
      input_dims is a tuple of integers containing the dimensions of the model
      input
      filters is a list containing the number of filters for each convolutional
      layer in the encoder, respectively
        the filters should be reversed for the decoder
      latent_dims is a tuple of integers containing the dimensions of the
      latent space representation
      Each convolution in the encoder should use a kernel size of (3, 3) with
      same padding and relu activation, followed by max pooling of size (2, 2)
      Each convolution in the decoder, except for the last two, should use a
      filter size of (3, 3) with same padding and relu activation, followed by
      upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as the
        number of channels in input_dims with sigmoid activation and
        no upsampling
    Returns: encoder, decoder, auto
      encoder is the encoder model
      decoder is the decoder model
      auto is the full autoencoder model
    """
    # Encoder
    encoder = keras.models.Sequential()

    encoder.add(keras.Input(shape=(input_dims)))

    # Hidden layers
    for i in range(len(filters)):
        encoder.add(keras.layers.Conv2D(filters=filters[i],
                                        kernel_size=(3, 3),
                                        padding="same",
                                        activation="relu"))
        encoder.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                              padding='same'))

    # Decoder
    decoder = keras.models.Sequential()

    decoder.add(keras.Input(shape=(latent_dims)))

    # Hidden Layers
    for i in range(len(filters) - 1, 0, -1):
        decoder.add(keras.layers.Conv2D(filters=filters[i],
                                        kernel_size=(3, 3),
                                        padding="same",
                                        activation="relu"))
        decoder.add(keras.layers.UpSampling2D((2, 2)))

    # Last hidden layer
    decoder.add(keras.layers.Conv2D(filters=filters[-1],
                                    kernel_size=(3, 3),
                                    padding="valid",
                                    activation="relu"))
    decoder.add(keras.layers.UpSampling2D((2, 2)))

    # Output layer
    decoder.add(keras.layers.Conv2D(filters=input_dims[2],
                                    kernel_size=(3, 3),
                                    padding="same",
                                    activation="sigmoid"))

    # Autoencoder
    inputs = keras.Input(shape=(input_dims))
    outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs,
                       outputs=outputs)

    auto.compile(optimizer='adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
