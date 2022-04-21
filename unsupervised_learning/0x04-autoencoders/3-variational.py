#!/usr/bin/env python3
"""File that contains the function autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder
    Args:
      input_dims is an integer containing the dimensions of the model input
      hidden_layers is a list containing the number of nodes for each hidden
      layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
      latent_dims is an integer containing the dimensions of the latent space
      representation
    Returns: encoder, decoder, auto
      encoder is the encoder model, which should output the latent
      representation, the mean, and the log variance, respectively
      decoder is the decoder model
      auto is the full autoencoder model
    """
    X_input = keras.Input(shape=(input_dims,))
    hidden_ly = keras.layers.Dense(units=hidden_layers[0], activation='relu')
    Y_prev = hidden_ly(X_input)
    for i in range(1, len(hidden_layers)):
        hidden_ly = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')
        Y_prev = hidden_ly(Y_prev)
    latent_ly = keras.layers.Dense(units=latent_dims, activation=None)
    z_mean = latent_ly(Y_prev)
    z_log_sigma = latent_ly(Y_prev)

    def sampling(args):
        """Sampling similar points in latent space"""
        z_m, z_stand_des = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_stand_des / 2) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean,
                                                          z_log_sigma])
    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma])

    X_decode = keras.Input(shape=(latent_dims,))
    hidden_ly_deco = keras.layers.Dense(units=hidden_layers[-1],
                                        activation='relu')
    Y_prev = hidden_ly_deco(X_decode)
    for j in range(len(hidden_layers) - 2, -1, -1):
        hidden_ly_deco = keras.layers.Dense(units=hidden_layers[j],
                                            activation='relu')
        Y_prev = hidden_ly_deco(Y_prev)
    last_ly = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_ly(Y_prev)
    decoder = keras.Model(X_decode, output)

    e_output = encoder(X_input)[-1]
    d_output = decoder(e_output)
    auto = keras.Model(X_input, d_output)

    def vae_loss(x, x_decoder_mean):
        x_loss = keras.backend.binary_crossentropy(x, x_decoder_mean)
        x_loss = keras.backend.sum(x_loss, axis=1)
        kl_loss = - 0.5 * keras.backend.mean(1 + z_log_sigma -
                                             keras.backend.square(z_mean) -
                                             keras.backend.exp(z_log_sigma),
                                             axis=-1)
        return x_loss + kl_loss

    auto.compile(loss=vae_loss, optimizer='adam')
    return encoder, decoder, auto
