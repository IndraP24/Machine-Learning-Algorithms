from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys


def shuffle_data(X, y, seed=None):
    """
    Random shuffle of the samples in X and y

    Parameter:

        X: Feature Matrix

        y: Labels

    Returns:
        Shuffled form of X and y
    """
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def normalize(X, axis=-1, order=2):
    """Normalize the dataset X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def standardize(X):
    """Standardize the dataset X"""
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X,mean(axis=0)) / X.std(axis=0)
    return X_std


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """
    Split the dataset into train and test sets

    Parameter:
        X: Feature Matrix

        y: Labels

        test_size: fractional size of the test dataset

        shuffle: If the dataset should be shuffled or not

    Returns:
         X_train, X_test, y_train and y_test
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test