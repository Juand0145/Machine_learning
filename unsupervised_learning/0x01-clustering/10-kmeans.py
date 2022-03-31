#!/usr/bin/env python3
"""File that contains the function kmeans"""
import sklearn.cluster


def kmeans(X, k):
    """
    function that performs K-means on a dataset
    Args:
      X is a numpy.ndarray of shape (n, d) containing the dataset
      k is the number of clusters
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each
    cluster
    clss is a numpy.ndarray of shape (n,) containing the index of the cluster
    in C that each data point belongs to
    """
    Kmean = sklearn.cluster.KMeans(n_clusters=k)
    Kmean.fit(X)

    C = Kmean.cluster_centers_
    clss = Kmean.labels_

    return C, clss
