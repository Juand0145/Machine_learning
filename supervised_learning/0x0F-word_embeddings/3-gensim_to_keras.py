#!/usr/bin/env python3
"""file that contains the function gensim_to_keras"""


def gensim_to_keras(model):
    """
    FUnction that converts a gensim word2vec model to a keras
    Embedding layer
    Args:
      model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """

    model = model.wv.get_keras_embedding(train_embeddings=False)
    return model
