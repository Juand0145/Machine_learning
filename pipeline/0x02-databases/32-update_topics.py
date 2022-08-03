#!/usr/bin/env python3
"""File that contains the function update_topics"""


def update_topics(mongo_collection, name, topics):
    """
    Function that changes all topics of a school document based on the name
    Args:
      mongo_collection will be the pymongo collection object
      name (string) will be the school name to update
      topics (list of strings) will be the list of topics approached in the school
    """
    mongo_collection.update_many({'name': name},
                                 {'$set': {'topics': topics}})
