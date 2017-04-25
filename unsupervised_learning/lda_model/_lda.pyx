#cython: language_level=2
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

# from cython.operator cimport  preincre

def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz, double[:] alpha, double[:] eta, double[:] rands):
    pass

cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double beta) nogil:
    pass