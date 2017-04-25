"""
    Latent Dirichlet Allocation using Collapsed Gibbs Sampling.
"""

from __future__ import absolute_import, division, unicode_literals
import logging
import sys

import utils.corpus_utils

import numpy as np

import _lda

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

    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        num_topics = self.num_topics
        num_iterations = self.num_iterations
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(num_topics))
        logger.info("n_iter: {}".format(num_iterations))

        self._num_zw = _nzw = np.zeros((num_topics, W), dtype=np.intc)
        self._num_dz = _ndz = np.zeros((D, num_topics), dtype=np.intc)
        self._num_z = _nz = np.zeros(num_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = utils.corpus_utils.matrix2lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % num_topics
            ZS[i] = z_new
            _ndz[d, z_new] += 1
            _nzw[z_new, w] += 1
            _nz[z_new] += 1
        self._loglikelihoods = []



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

    """
         Applying dimensionality reduction on X
         
         Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions
            
    """
    def fit_transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = np.atleast_2d(X)
        self._fit(X)
        return self._doc_topic

    """
        Transform the data X according to previously fitted model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudo_count estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

    """
    def transform(self, X, max_iter=20, tol=1e-16):
        if isinstance(X, np.ndarray):
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.num_topics))
        WS, DS = utils.corpus_utils.matrix2lists(X)

        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS==d], max_iter, tol)

        return doc_topic

    """
        Transform a single document according to the previously fit model
        
        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document
            
    """
    def _transform_single(self, doc, max_iter, tol):
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1):  # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis]  # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()

        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)

        return theta_doc

    """
        Fit the model to data X
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
            
    """
    def _fit(self, X):
        random_state = utils.corpus_utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X)

        for iter in range(self.num_iterations):
            random_state.shuffle(rands)
            if iter%self.refresh == 0:
                ll = self.log_likelihood()
                logger.info("<{}> log likelihood: {%.0f}".format(iter, ll))
                self._loglikelihoods.append(ll)
            self._sample_topics(rands)

        ll = self.log_likelihood()
        logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))

        self._components = (self._num_zw+self.beta).astype(float)
        self._components /= np.sum(self._components, axis=1)[:, np.newaxis]
        self._topic_word = self._components

        self._doc_topic = (self._num_dz + self.alpha).astype(float)
        self._doc_topic /= np.sum(self._doc_topic, axis=1)[:, np.newaxis]

        del self.WS
        del self.DS
        del self.ZS

        return self

    # Calculate complete log likelihood, log p(w,z)
    def log_likelihood(self):
        nzw, ndz, nz= self._num_zw, self._num_dz, self._num_z
        alpha = self.alpha
        beta = self.beta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        # call c function via _lda.pyx
        return _lda._loglikelihood(nzw, ndz, nz, nd, alpha, beta)

    # Samples all topic assignments. Called once per iteration.
    def _sample_topics(self, rands):
        num_topics, vocabulary_size = self._num_zw.shape
        alpha = np.repeat(self.alpha, num_topics).astype(np.float64)
        beta = np.repeat(self.beta, vocabulary_size).astype(np.float64)
        _lda._sample_topics(self.WS, self.DS, self.ZS, self._num_zw, self._num_dz, self._num_z, alpha, beta, rands)
        



