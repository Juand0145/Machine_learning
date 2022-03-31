#!/usr/bin/env python3
"""File that contains the function gmm"""
import sklearn.mixture


def gmm(X, k):
    """
    Function that calculates a GMM from a dataset
    Args:
      X is a numpy.ndarray of shape (n, d) containing the dataset
      k is the number of clusters
    Returns: pi, m, S, clss, bic
      pi is a numpy.ndarray of shape (k,) containing the cluster priors
      m is a numpy.ndarray of shape (k, d) containing the centroid means
      S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices
      clss is a numpy.ndarray of shape (n,) containing the cluster indices
      for each data point
      bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
      value for each cluster size tested
    """
    gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    gm = gm.fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_

    clss = gm.predict(X)
    bic = gm.bic(X)

    return pi, m, S, clss, bic
