#!/usr/bin/env python3
"""File that contains the fucntion"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram BLEU score for a sentence
    Args:
      references is a list of reference translations
      each reference translation is a list of the words in the translation
      sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    sentence_length = len(sentence)
    references_length = []
    words = {}

    for translation in references:
        references_length.append(len(translation))
        for word in translation:
            if word in sentence and word not in words.keys():
                words[word] = 1

    total = sum(words.values())
    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    best_match = len(references[index])

    if sentence_length > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(sentence_length))
    BLEU_score = BLEU * np.exp(np.log(total / sentence_length))

    return BLEU_score
