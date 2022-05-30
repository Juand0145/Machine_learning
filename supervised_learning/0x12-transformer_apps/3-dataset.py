#!/usr/bin/env python3
"""File that contains the class Dataset"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """Class that loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """
        Class constructor
        creates the instance attributes:
          data_train, which contains the ted_hrlr_translate/pt_to_en tf.data.
          Dataset train split, loaded as_supervided
          data_valid, which contains the ted_hrlr_translate/pt_to_en tf.data.
          Dataset validate split, loaded as_supervided
          tokenizer_pt is the Portuguese tokenizer created from the training
          tokenizer_en is the English tokenizer created from the training set
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples["train"]
        self.data_valid = examples["validation"]

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        def filter_max_length(x, y, max_len=max_len):
            """Funcion that filter the max lenght"""
            filter = tf.logical_and(tf.size(x) <= max_len,
                                    tf.size(y) <= max_len)
            return filter

        # Set attributes to encoded data
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Filter training and validation by max_len number of tokens
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_valid = self.data_valid.filter(filter_max_length)

        # Cache the dataset to memory
        self.data_train = self.data_train.cache()

        # Shuffle the training dataset
        shuff = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shuff)

        # Split training and validation datasets into padded batches
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

        # Increase performance by prefetching training dataset
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

    def tokenize_dataset(self, data):
        """
        Public instance method that creates sub-word tokenizers for our dataset
        Args:
            data is a tf.data.Dataset whose examples are formatted as
            a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
            The maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
          tokenizer_pt is the Portuguese tokenizer
          tokenizer_en is the English tokenizer
        """
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for en, pt in data), target_vocab_size=2**15
        )
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15
        )
        return token_pt, token_en

    def encode(self, pt, en):
        """
        Public instance method that encodes a translation into tokens
        Args:
          pt is the tf.Tensor containing the Portuguese sentence
          en is the tf.Tensor containing the corresponding English sentence
          The tokenized sentences should include the start and end of sentence
          tokens
          The start token should be indexed as vocab_size
          The end token should be indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
          pt_tokens is a np.ndarray containing the Portuguese tokens
          en_tokens is a np.ndarray. containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Public instance method that  acts as a tensorflow wrapper for
        the encode instance method
        """
        wrapped_pt, wrapped_en = tf.py_function(self.encode,
                                                [pt, en],
                                                [tf.int64, tf.int64])
        wrapped_pt.set_shape([None])
        wrapped_en.set_shape([None])

        return wrapped_pt, wrapped_en
