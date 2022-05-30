#!/usr/bin/env python3
"""File that contains the class Dataset"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """Class that loads and preps a dataset for machine translation"""

    def __init__(self):
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
