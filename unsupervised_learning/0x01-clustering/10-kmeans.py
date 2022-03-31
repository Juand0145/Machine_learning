#!/usr/bin/env python3
"""File that contains the function kmeans"""
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    function that performs K-means on a dataset
    Args:
      X is a numpy.ndarray of shape (n, d) containing the dataset
      k is the number of clusters
      The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each
    cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster
    in C that each data point belongs to
    """
    Kmean = KMeans(n_clusters=5)
    Kmean.fit(X)

    C = Kmean.cluster_centers_
    clss = Kmean.labels_

    return C, clss
