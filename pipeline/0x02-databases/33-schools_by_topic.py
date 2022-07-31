#!/usr/bin/env python3
"""Fle that contains the function schools_by_topic"""


def schools_by_topic(mongo_collection, topic):
    """
    Function that  returns the list of school having a specific topic
    Args:
      mongo_collection will be the pymongo collection object
      topic (string) will be topic searched
    """
    schools = []
    documents = mongo_collection.find({'topics': {'$all': [topic]}})
    for doc in documents:
        schools.append(doc)
    return schools
