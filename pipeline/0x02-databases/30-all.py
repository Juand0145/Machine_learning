#!/usr/bin/env python3
"""File that contains the function list_all"""


def list_all(mongo_collection):
    """
    Is a python function thatlists all documents in a collection
    Args:
      mongo_collection will be the pymongo collection object
     Return:
      an empty list if no document in the collection
    """
    all_docs = []
    collection = mongo_collection.find()
    for document in collection:
        all_docs.append(document)
    return all_docs
