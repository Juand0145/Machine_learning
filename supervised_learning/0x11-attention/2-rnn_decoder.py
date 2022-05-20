#!/usr/bin/env python3
"""File that contains the class RNNDecoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder (tf.keras.layers.Layer):
    """Class that decode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
          Args:
          vocab is an integer representing the size of the output vocabulary
          embedding is an integer representing the dimensionality of the
          embedding vector
          units is an integer representing the number of hidden units in
          the RNN cell
          batch is an integer representing the batch size
        Sets the following public instance attributes:
          embedding - a keras Embedding layer that converts words from the
          vocabulary into an embedding vector
          gru - a keras GRU layer with units units
          Should return both the full sequence of outputs as well as the last
          hidden state
          Recurrent weights should be initialized with glorot_uniform
          F - a Dense layer with vocab units
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Args:
          x is a tensor of shape (batch, 1) containing the previous word in the
          target sequence as an index of the target vocabulary
          s_prev is a tensor of shape (batch, units) containing the previous
          decoder hidden state
          hidden_states is a tensor of shape (batch, input_seq_len, units)
          ontaining the outputs of the encoder
          You should concatenate the context vector with x in that order
        Returns: y, s
          y is a tensor of shape (batch, vocab) containing the output word
          as a one hot vector in the target vocabulary
          s is a tensor of shape (batch, units) containing the new decoder
          hidden state
        """
        batch, units = s_prev.shape
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)
        concat_input = tf.concat([tf.expand_dims(context, 1),
                                  embeddings],
                                 axis=-1)

        # passing the concatenated vector to the GRU
        outputs, hidden = self.gru(concat_input)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)
        return y, hidden
