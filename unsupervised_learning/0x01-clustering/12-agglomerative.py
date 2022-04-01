#!/usr/bin/env python3
"""File that contains the function agglomerative"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Function that performs agglomerative clustering on a dataset
    Args:
      X is a numpy.ndarray of shape (n, d) containing the dataset
      dist is the maximum cophenetic distance for all clusters
      Performs agglomerative clustering with Ward linkage
      Displays the dendrogram with each cluster displayed in a different color
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster indices for each data point
    """
    Z = linkage(X, 'ward')
    plt.figure(figsize=(15, 10))
    dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = fcluster(Z, t=dist, criterion="distance")
    plt.figure(figsize=(15, 10))

    return clss
