"""
    Latent Dirichlet Allocation using Collapsed Gibbs Sampling.
"""
from __future__ import absolute_import, division, unicode_literals
import logging
import sys

import utils.corpus_utils

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange

class LDA:
    def __init__(self):
        pass