from __future__ import division
from gensim.models import ldamodel
from gensim.matutils import dirichlet_expectation
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma

import six
from six.moves import xrange
from itertools import chain

from utils import corpus_handle

import numpy as np

import logging

# Latent Dirichlet Allocation + Collapsed Gibbs Sampling

lda = ldamodel.LdaModel();

logger = logging.getLogger('gensim.models.ldamodel')

# Updates a given prior using Newton's method.
def update_dirichlet_prior(prior, N, logphat, rho):
    dprior = np.copy(prior)
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior > 0):
        prior += rho * dprior
    else:
        logger.warning("updated prior not positive")

    return prior

class LDAState():
    def __init__(self, eta, shape):
        self.eta = eta
        self.s_stats = np.zeros(shape)
        self.num_docs = 0

class LDA():
    def __init__(self, corpus=None, num_topics=100, id2word=None, distributed=None, chunk_size=2000, passes=1,
                 update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, evaluate_every=10, iterations=200,
                 gamma_threshold=0.001, min_prob=0.01, random_state=None, ns_conf={}, min_phi_val=0.01, per_word_topics=False):
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('At least one of corpus/id2word must be specified.')

        if self.id2word is None:
            logger.warning('No word-id mapping provided; initializing from corpus, assuming identity')
            self.id2word = corpus_handle.dict_from_corpus(corpus)
            self.num_items = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_items = 1 + max(self.id2word.keys())
        else:
            self.num_items = 0

        if self.num_items == 0:
            raise ValueError(" Cannot compute LDA over an empty collection(no items)")

        self.distributed = distributed
        self.num_topics = num_topics
        self.chunk_size = chunk_size
        self.decay = decay
        self.offset = offset
        self.min_prob = min_prob
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every
        self.evaluate_every = evaluate_every
        self.min_phi_val = min_phi_val

        self.alpha, self.optimize_alpha = self.init_dirichlet_prior(alpha, 'alpha')

        assert self.alpha.shape == (self.num_topics, ), "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dirichlet_prior(eta, 'eta')

        self.random_state = corpus_handle.get_random_state(random_state)

        assert (self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms)), (
                "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
                (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms))

        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        if not distributed:
            logger.info("Using serial LDA version on this node.")
            self.dispatcher = None
            self.num_workers = 1
        else:
            pass

        self.state = LDAState(self.eta, (self,num_topics, self.num_items))
        self.state.s_stats = self.random_state.gamma(100., 1./100., (self.num_topics, self.num_items))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.s_stats))

        # if a training corpus was provided, start training estimating right away.
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, chunk_as_numpy=use_numpy)


    def init_dirichlet_prior(self, prior, name):
        pass

    def update(self, corpus, chunk_size=None, decay=None, offset=None, passes=None, update_every=None,
               evaluate_every=None, iterations=None, gamma_threshold=None, chunk_as_numpy=False):
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if evaluate_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold

        try:
            len_corpus = len(corpus)
        except:
            logger.warning("Input corpus stream has no len(), counting documents")
            len_corpus = sum(1 for _ in corpus)

        if len_corpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return

        if chunk_size is None:
            chunk_size = min(len_corpus, self.chunk_size)

        self.state.num_docs += len_corpus

        if update_every:
            update_type = 'online'
            update_after = min(len_corpus, update_every*self.num_workers*chunk_size)
        else:
            update_type = 'batch'
            update_after = len_corpus
        evaluate_after = min(len_corpus, (evaluate_every or 0)*self.num_workers*chunk_size)

        update_per_pass = max(1, len_corpus/update_after)

        # if update_per_pass*passes < 10:

        def rho():
            return pow(offset+pass_+(self.num_updates/chunk_size), -decay)

        for pass_ in xrange(passes):

            other = LDAState(self.eta, self.state.s_stats.shape)

            dirty = False


