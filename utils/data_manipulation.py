from __future__ import division
import numpy as np
import math
import sys

def shuffle_data(X, y, seed=None):
    # Concatenate X and y and shuffle
    X_y = np.concatenate((X, y.reshape(1, len(y)).T), axis=1);
    if seed:
        np.random.seed(seed)
    np.random.shuffle(X_y)
    X = X_y[:, :-1]
    y = X_y[:, -1].astype(int)

    return X, y

def divide_on_feature(X, feature_i, threshold):
    pass

def get_random_subset(X, y, n_subsets, replacements=True):
    pass

def normalize(X, axis=-1, order=2):
    pass

def standardize(X):
    pass

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    pass

def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    pass

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

