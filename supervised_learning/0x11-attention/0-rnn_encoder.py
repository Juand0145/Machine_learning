#!/usr/bin/env python3
"""File that contains the Class RNNEncoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Class that works to encode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class contructor, Args:
          vocab is an integer representing the size of the input vocabulary
          embedding is an integer representing the dimensionality of the
          embedding vector
          units is an integer representing the number of hidden units in
          the RNN cell
          batch is an integer representing the batch size
        Sets the following public instance attributes:
          batch - the batch size
          units - the number of hidden units in the RNN cell
          embedding - a keras Embedding layer that converts words from the
          vocabulary into an embedding vector
          gru - a keras GRU layer with units units
            Should return both the full sequence of outputs as well as the last
            hidden state
            Recurrent weights should be initialized with glorot_uniform
        """
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

        super(RNNEncoder, self).__init__()

    def initialize_hidden_state(self):
        """
        Public instance method that Initializes the hidden states for the
        RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units)containing the initialized
        hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        hidden_state = initializer(shape=(self.batch, self.units))

        return hidden_state

    def call(self, x, initial):
        """
        Args:
          x is a tensor of shape (batch, input_seq_len) containing the input
          to the encoder layer as word indices within the vocabulary
          initial is a tensor of shape (batch, units) containing the initial
          hidden state
        Returns: outputs, hidden
          outputs is a tensor of shape (batch, input_seq_len, units)containing
          the outputs of the encoder
          hidden is a tensor of shape (batch, units) containing the last hidden
          state of the encoder
        """
        embedding = self.embedding(x)
        outputs, last_hiddenQ = self.gru(embedding,
                                         initial_state=initial)
        return outputs, last_hiddenQ
