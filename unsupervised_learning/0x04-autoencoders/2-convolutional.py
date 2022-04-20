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
    input_encoder = keras.Input(shape=(input_dims))

    prev_layer = input_encoder
    for i in range(len(filters)):
        hidden_layer = keras.layers.Conv2D(filters[i],
                                           activation='relu',
                                           kernel_size=(3, 3),
                                           padding='same')
        prev_layer = hidden_layer(prev_layer)
        pooling_layer = keras.layers.MaxPooling2D((2, 2),
                                                  padding='same')
        prev_layer = pooling_layer(prev_layer)
    encoder_outputs = prev_layer
    encoder = keras.Model(inputs=input_encoder, outputs=encoder_outputs)

    # decoder
    input_decoder = keras.Input(shape=(latent_dims))

    prev_layer = input_decoder
    for i in range(len(filters) - 1, 0, -1):
        hidden_layer = keras.layers.Conv2D(filters[i],
                                           activation='relu',
                                           kernel_size=(3, 3),
                                           padding='same')
        prev_layer = hidden_layer(prev_layer)
        upsample_layer = keras.layers.UpSampling2D((2, 2))
        prev_layer = upsample_layer(prev_layer)

    # Decoder Last layer
    last_layer = keras.layers.Conv2D(filters[0],
                                     kernel_size=(3, 3),
                                     padding='valid',
                                     activation='relu')
    prev_layer = last_layer(prev_layer)
    upsample_layer = keras.layers.UpSampling2D((2, 2))
    prev_layer = upsample_layer(prev_layer)

    # Decoder Last Layer
    output_layer = keras.layers.Conv2D(input_dims[2],
                                       activation='sigmoid',
                                       kernel_size=(3, 3),
                                       padding='same')
    decoder_outputs = output_layer(prev_layer)
    decoder = keras.Model(inputs=input_decoder, outputs=decoder_outputs)

    # autoencoder
    inputs = input_encoder
    outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=outputs)
    auto.compile(optimizer='adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
