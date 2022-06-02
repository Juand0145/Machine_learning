#!/usr/bin/env python3
"""File that conatins the function question_answer"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import os
import numpy as np

def question_answer(corpus_path):
  """
  Function that answers questions from multiple reference texts
  Args:
    corpus_path is the path to the corpus of reference documents
  """
  tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  model_BERT = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
  model_semantic = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

  while True:
    question = input("Q: ").lower()
    bye = ["exit", "quit", "goodbye", "bye"]

    if question in bye:
      print(f"A: Goodbye")
      break
    else:
      # Semantic Correlation
      documents = [question]

      # If the text is inside a different termination than md change this
      for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
                continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
                documents.append(f.read())

      embeddings = model_semantic(documents)

      correlations = np.inner(embeddings, embeddings)

      closest = np.argmax(correlations[0, 1:])

      similar = documents[closest + 1]

      # Bert Question response     
      paragraph = similar

      paragraph_tokens = tokenizer.tokenize(paragraph)
      question_tokens = tokenizer.tokenize(question)
      
      tokens = ["CLS"] + question_tokens +["SEP"] + paragraph_tokens +["SEP"]
      input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_word_ids)
      input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

      input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
      tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
      outputs = model_BERT([input_word_ids, input_mask, input_type_ids])

      # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
      short_start = tf.argmax(outputs[0][0][1:]) + 1
      short_end = tf.argmax(outputs[1][0][1:]) + 1
      answer_tokens = tokens[short_start: short_end + 1]
      answer = tokenizer.convert_tokens_to_string(answer_tokens)

      if answer is None or answer is "" or question in answer:
        print("A: Sorry, I do not understand your question.")
      else:
        print(f"A: {answer}")
