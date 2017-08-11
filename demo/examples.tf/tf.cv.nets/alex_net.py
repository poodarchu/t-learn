from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alexnet_v2_arg_scope(weight_decay=0.0005):
    pass