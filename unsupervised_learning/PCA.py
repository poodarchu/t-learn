# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

from utils.data_operate import variance_matrix, correlation_matrix
from utils.data_manipulate import standardize

'''
    The main purpose of PCA is the analysis of data to identify patterns and finding patterns
    to reduce the dimension of the data set with minimal loss of information.
    Namely, the desired outcome is to project a feature space(our data set consisting of n-dimensional
    samples) onto a smaller subspace that could represent our data set 'well'.
'''

class PCA():

    def __init__(self): pass

    def transform(self, X, n_components):
        pass

    def get_color_map(self, N):
        pass

    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_lables=None):
        pass

    def plot_in_3d(self, X, y=None):
        pass

if __name__ == "__main__":
    pass