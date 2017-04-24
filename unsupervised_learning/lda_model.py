"""
    Latent Dirichlet Allocation using Collapsed Gibbs Sampling.
"""

from __future__ import absolute_import, division, unicode_literals
import logging
import sys

import utils.corpus_utils

logger = logging.getLogger('lda_model')

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange


"""
    Introduction: 
    --------------------
    In natural language processing, latent Dirichlet allocation (LDA) is a generative statistical model 
    that allows sets of observations to be explained by unobserved groups that explain why some parts of 
    the data are similar. For example, if observations are words collected into documents, it posits that 
    each document is a mixture of a small number of topics and that each word's creation is attributable to 
    one of the document's topics.
    
    Parameters:
    --------------------
    num_topics: int
        Number of topics.
        
    num_iterations: int, default=2000
        Number of sampling iterations.
        
    alpha: float, default=0.1
        Dirichlet parameter for distribution over topics.
        
    beta: float, default=0.01
        Dirichlet parameter for distribution over words.
        
    random_state: int / RandomState, optional
        The generator used for the initial topics.
        
    Attributes:
    --------------------
    _component: Array, shape=[num_topics, num_features]
        Point estimate of the topic-word distributions (Phi in literature).
    _num_zw: Array, shape=[num_topics, num_features]
        Matrix of counts recording topic-word assignments in final iteration.
    _topic_word: 
        Alias for _component.
    _num_dz: Array, shape=[num_samples, num_topics]
        Matrix of counts recording document-topic assignments in final iteration. 
    _doc_topic: Array, shape=[num_samples, num_features]
        Point estimate of the doc-topic distributions (Theta in literature).
    _num_z: Array, shape=[num_topics]
        Array of topic assignment count in final iteration.
    
"""

class LDA(object):

    def __init__(self, num_topics, num_iterations=2000, alpha=0.1, beta=0.01, random_state=None, refresh=10):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

        self.random_state = random_state
        self.refresh = refresh

        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be great than 0, 0.1 and 0.01 are defaults respectively.")

        # random numbers that are reused.
        rng = utils.corpus_utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1 MB of random variates

        # Configure console logging if not already configured.
        if (len(logger.handlers) == 1) and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)

    """ Fit the model with X.
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        self : object
            Returns the instance itself.
    """
    def fit(self, X, y=None):

        self._fit(X)
        return self


