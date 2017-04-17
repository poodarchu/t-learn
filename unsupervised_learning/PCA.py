from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

from utils.data_operate import variance_matrix, correlation_matrix
from utils.data_manipulate import standardize

class PCA():
    def __init__(self): pass