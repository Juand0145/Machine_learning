#!/usr/bin/env python3
"""Function that hat answers questions from a reference text"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def answer_loop(reference):
  """
  Function that hat answers questions from a reference text
  Args:
    reference is the reference text
  """
  bye = ["exit", "quit", "goodbye", "bye"]

  tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
  
  paragraph = reference
  paragraph_tokens = tokenizer.tokenize(paragraph)

  while True:
    question = input("Q: ").lower()

    if question in bye:
      print(f"A: Goodbye")
      break
    else:
      question_tokens = tokenizer.tokenize(question)
      
      tokens = ["CLS"] + question_tokens +["SEP"] + paragraph_tokens +["SEP"]
      input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_word_ids)
      input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

      input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
      tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
      outputs = model([input_word_ids, input_mask, input_type_ids])

      # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
      short_start = tf.argmax(outputs[0][0][1:]) + 1
      short_end = tf.argmax(outputs[1][0][1:]) + 1
      answer_tokens = tokens[short_start: short_end + 1]
      answer = tokenizer.convert_tokens_to_string(answer_tokens)

      if answer is None or answer is "" or question[0] in answer:
        print("A: Sorry, I do not understand your question.")
      else:
        print(f"A: {answer}")
      