from __future__ import division
import numpy as np
import math
import sys


# Concatenate X and y and shuffle
def shuffle_data(X, y, seed=None):
    X_y = np.concatenate((X, y.reshape(1, len(y)).T), axis=1);
    if seed:
        np.random.seed(seed)
    np.random.shuffle(X_y)
    X = X_y[:, :-1]
    y = X_y[:, -1].astype(int)

    return X, y


# Divide data set based on whether feature_i's value is larger than the given threshold
# Return split-ed data set concatenate together
def divide_on_feature(X, feature_i, threshold):
    split_func = None
    if (isinstance(threshold, int) or isinstance(threshold, float)):
        split_func = lambda sample : sample[feature_i] >= threshold
    else:
        split_func = lambda sample : sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


# Return random subset (with replacements) of the data
def get_random_subset(X, y, n_subsets, replacements=True):
    n_samples = np.shape(X)[0]
    X_y = np.concatenate((X, y.reshape(1, len(y)).T), axis=1)
    np.random.shuffle(X_y)

    subset = []

    # Use 70% of training samples without replacements
    n_subsamples = (n_samples * 0.7).astype(int)
    if replacements:
        n_subsamples = n_samples

    for _ in range(n_subsets):
        idx = np.random.choice(range(n_samples), size=np.shape(range(n_subsamples)), replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subset.append([X, y])

    return subset


# Normalize the data set X
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1

    return X / np.expand_dims(l2, axis)


# Standardize the data set X
def standardize(X):
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    for col in range(np.shape(X), 1):
        if std[col]:
            X_std[ : col] = (X_std[ : col] - mean[col]) / std[col]

    return X_std


# Split data set into Train and Test
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1/test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


# Split data set into k sets of training/test data
def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = len(y)

    left_overs = {}
    n_left_overs = (n_samples % k)

    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X, k)
    y_split = np.split(y, k)

    sets = []

    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i + 1:], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)

    return np.array(sets)


# Convert an array of nominal values into a binary matrix
def categorical_to_binary(x):
    n_col = np.amax(x) + 1
    ret_binary = np.zeros(len(x), n_col)
    for i in range(len(x)):
        ret_binary[i, x[i]] = 1

    return ret_binary

# Convert from binary vectors to normal values
def binary_to_categorical(x):
    ret_categorical = []
    len = len(x)
    for i in range(len):
        if not 1 in x[i]:
            ret_categorical.append(0)
        else:
            i_where_one = np.where(x[i] == 1)[0][0]
            ret_categorical.append(i_where_one)

    return ret_categorical

# Converts a vector into a diagonal matrix
def make_diagonal(x):
    l = len(x);
    ret = np.zeros(l, l)
    for i in range(l):
        ret[i, i] = x[i]

    return ret

