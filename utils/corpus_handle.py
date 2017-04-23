import logging, warnings

import re
import unicodedata
import random
import itertools
import tempfile
import functools
import multiprocessing
import shutil

import numpy as np
import numbers
import scipy.sparse

from six import iterkeys, iteritems, u, string_types, unichr
from six.moves import xrange

def get_max_id(corpus):
    maxid = -1
    for document in corpus:
        maxid = max(maxid, max([-1] + [field_id for field_id, _ in document]))

        return maxid

'''
Objects of this class act as dictionaries that map integer->string(integer), 
for a specified range of integers <0, num_items)

This is meant to avoid allocating real dictionaries when 'num_terms' is huge, which
 is a waste of memory.
'''
class FakeDict(object):
    def __init__(self, num_terms):
        self.num_terms = num_terms

    def __str__(self):
        return "FakeDict(num_terms=%s" % self.num_terms

    



def dict_from_corpus(corpus):
    num_terms = 1 + get_max_id(corpus)
    id2word = FakeDict(num_terms)

    return id2word