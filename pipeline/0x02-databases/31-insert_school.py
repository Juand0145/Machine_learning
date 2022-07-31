#!/usr/bin/env python3
"""File that contains the function insert_school"""


def insert_school(mongo_collection, **kwargs):
    """
    Funnction that inserts a new document in a collection based on kwargs
    Args:
      mongo_collection will be the pymongo collection object
    Returns the new _id
    """
    document = mongo_collection.insert_one(kwargs)
    return (document.inserted_id)
