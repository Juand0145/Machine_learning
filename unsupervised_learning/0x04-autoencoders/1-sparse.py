#!/usr/bin/env python3
"""File that contains the function autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Is a function that creates a sparse autoencoder
    Args:
      input_dims is an integer containing the dimensions of the model input
      hidden_layers is a list containing the number of nodes for each hidden
      layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
      latent_dims is an integer containing the dimensions of the latent space
      representation
      lambtha is the regularization parameter used for L1 regularization on the
      encoded output
      Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model
    """
    # Encoder
    encoder = keras.models.Sequential()

    encoder.add(keras.Input(shape=(input_dims, )))

    number_layers = len(hidden_layers)
    for i in range(number_layers - 1, -1, -1):
        encoder.add(keras.layers.Dense(units=hidden_layers[i],
                                       activation="relu"))

    regularizer = keras.regularizers.l1(lambtha)

    encoder.add(keras.layers.Dense(units=latent_dims,
                                   activation="relu",
                                   activity_regularizer=regularizer))

    # Decoder
    decoder = keras.models.Sequential()

    decoder.add(keras.Input(shape=(latent_dims, )))

    number_layers = len(hidden_layers)
    for i in range(number_layers - 1, -1, -1):
        decoder.add(keras.layers.Dense(units=hidden_layers[i],
                                       activation="relu"))

    decoder.add(keras.layers.Dense(units=input_dims,
                                   activation="sigmoid"))

    # Autoencoder
    input = keras.Input(shape=(input_dims, ))
    output = decoder(encoder(input))
    auto = keras.Model(inputs=input,
                       outputs=output)

    auto.compile(optimizer="adam",
                 loss="binary_crossentropy")

    return encoder, decoder, auto
