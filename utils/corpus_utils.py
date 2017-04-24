from __future__ import absolute_import, unicode_literals
import logging
import numbers
import sys

import numpy as np

PY2 = sys.version_info[0] == 2
if PY2:
    import itertools
    zip = itertools.izip

logger = logging.getLogger('LDA')

def check_random_state(seed):
    pass

def matrix2lists(doc_word):
    pass

def lists2matrix(WS, DS):
    pass

def dtm2ldac(dtm, offset=0):
    pass

def ldac2dtm(stream, offset=0):
    pass



