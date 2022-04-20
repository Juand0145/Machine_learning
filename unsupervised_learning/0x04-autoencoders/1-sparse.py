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
    input_encoder = keras.Input(shape=(input_dims, ))

    prev_layer = input_encoder
    number_layers = len(hidden_layers)
    for i in range(number_layers):
        hidden_layer = keras.layers.Dense(units=hidden_layers[i],
                                          activation="relu")

        prev_layer = hidden_layer(prev_layer)

    regularizer = keras.regularizers.l1(lambtha)
    output_layer = keras.layers.Dense(units=latent_dims,
                                      activation="relu",
                                      activity_regularizer=regularizer)

    output_encoder = output_layer(prev_layer)

    encoder = keras.Model(inputs=input_encoder,
                          outputs=output_encoder)

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))

    prev_later = input_decoder
    for i in range(number_layers - 1, -1, -1):
        hidden_layer = keras.layers.Dense(units=hidden_layers[i],
                                          activation='relu')
        prev_layer = hidden_layer(prev_later)

    output_layer = keras.layers.Dense(units=input_dims,
                                      activation='sigmoid')

    decoder_outputs = output_layer(prev_layer)

    decoder = keras.Model(inputs=input_decoder,
                          outputs=decoder_outputs)

    # Autoencoder
    inputs = input_encoder
    outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs,
                       outputs=outputs)
    auto.compile(optimizer='adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
