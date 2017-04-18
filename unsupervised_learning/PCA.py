# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from utils.data_operate import variance_matrix, correlation_matrix
from utils.data_manipulate import standardize

import logging
import unittest
import math
import numpy as np
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
from scipy.sparse import coo_matrix, issparse, spdiags

"""
Functions for principal component analysis (PCA) and accuracy checks
---------------------------------------------------------------------
This module contains eight functions:
pca
    principal component analysis (singular value decomposition)
eigens
    eigendecomposition of a self-adjoint matrix
eigenn
    eigendecomposition of a nonnegative-definite self-adjoint matrix
diffsnorm
    spectral-norm accuracy of a singular value decomposition
diffsnormc
    spectral-norm accuracy of a centered singular value decomposition
diffsnorms
    spectral-norm accuracy of a Schur decomposition
mult
    default matrix multiplication
set_matrix_mult
    re-definition of the matrix multiplication function "mult"
---------------------------------------------------------------------
"""

def diffsnorm(A, U, s, Va, n_iter=20):
    (m, n) = A.shape
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    if m >= n:
        if isreal:
            x = np.random.normal(size=(n, 1))
        else:
            x = np.random.normal(size=(n, 1)) \
                + 1j * np.random.normal(size=(n, 1))

        x = x / norm(x)

        # Run n_iter iterations of the power method.
        for it in range(n_iter):
            # Set y = (A - U diag(s) Va)x.
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))
            # Set x = (A' - Va' diag(s)' U')y.
            x = mult(y.conj().T, A).conj().T - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

            # Normalize x, memorizing its Euclidean norm.
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:
        # Generate a random vector y.
        if isreal:
            y = np.random.normal(size=(m, 1))
        else:
            y = np.random.normal(size=(m, 1)) + 1j * np.random.normal(size=(m, 1))

        y = y / norm(y)

        # Run n_iter iterations of the power method.
        for it in range(n_iter):
            # Set x = (A' - Va' diag(s)' U')y.
            x = mult(y.conj().T, A).conj().T - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            # Set y = (A - U diag(s) Va)x.
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))

            # Normalize y, memorizing its Euclidean norm.
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


def diffsnormc(A, U, s, Va, n_iter=20):
    (m, n) = A.shape
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and \
            np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    # Calculate the average of the entries in every column.
    c = A.sum(axis=0) / m
    c = c.reshape((1, n))

    if m >= n:
        # Generate a random vector x.
        if isreal:
            x = np.random.normal(size=(n, 1))
        else:
            x = np.random.normal(size=(n, 1)) + 1j * np.random.normal(size=(n, 1))

        x = x / norm(x)

        # Run n_iter iterations of the power method.
        for it in range(n_iter):
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            y = mult(A, x) - np.ones((m, 1)).dot(c.dot(x)) - U.dot(np.diag(s).dot(Va.dot(x)))
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            x = mult(y.conj().T, A).conj().T \
                - c.conj().T.dot(np.ones((1, m)).dot(y)) \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

            # Normalize x, memorizing its Euclidean norm.
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:
        # Generate a random vector y.
        if isreal:
            y = np.random.normal(size=(m, 1))
        else:
            y = np.random.normal(size=(m, 1)) + 1j * np.random.normal(size=(m, 1))

        y = y / norm(y)

        # Run n_iter iterations of the power method.
        for it in range(n_iter):
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            x = mult(y.conj().T, A).conj().T \
                - c.conj().T.dot(np.ones((1, m)).dot(y)) \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            y = mult(A, x) - np.ones((m, 1)).dot(c.dot(x)) - U.dot(np.diag(s).dot(Va.dot(x)))

            # Normalize y, memorizing its Euclidean norm.
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


def diffsnorms(A, S, V, n_iter=20):
    (m, n) = A.shape
    (m2, k) = V.shape
    (k2, k3) = S.shape

    assert m == n
    assert m == m2
    assert k == k2
    assert k2 == k3

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(V) and np.isrealobj(S):
        isreal = True
    else:
        isreal = False

    # Generate a random vector x.
    if isreal:
        x = np.random.normal(size=(n, 1))
    else:
        x = np.random.normal(size=(n, 1)) + 1j * np.random.normal(size=(n, 1))

    x = x / norm(x)

    # Run n_iter iterations of the power method.
    for it in range(n_iter):
        # Set y = (A-VSV')x.
        y = mult(A, x) - V.dot(S.dot(V.conj().T.dot(x)))
        # Set x = (A'-VS'V')y.
        x = mult(y.conj().T, A).conj().T - V.dot(S.conj().T.dot(V.conj().T.dot(y)))

        # Normalize x, memorizing its Euclidean norm.
        snorm = norm(x)
        if snorm == 0:
            return 0
        x = x / snorm

    snorm = math.sqrt(snorm)

    return snorm


def eigenn(A, k=6, n_iter=4, l=None):
    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Check whether A is self-adjoint to nearly the machine precision.
    x = np.random.uniform(low=-1.0, high=1.0, size=(n, 1))
    y = mult(A, x)
    z = mult(x.conj().T, A).conj().T
    assert (norm(y - z) <= .1e-11 * norm(y)) and (norm(y - z) <= .1e-11 * norm(z))

    # Eigendecompose A directly if l >= n/1.25.
    if l >= (n / 1.25):
        (d, V) = eigh(A.todense() if issparse(A) else A)
        # Retain only the entries of d with the k greatest absolute
        # values and the corresponding columns of V.
        idx = abs(d).argsort()[-k:][::-1]
        return abs(d[idx]), V[:, idx]

    # Apply A to a random matrix, obtaining Q.
    if isreal:
        R = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
    if not isreal:
        R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l))

    Q = mult(A, R)

    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    if n_iter == 0:
        anorm = 0
        for j in range(l):
            anorm = max(anorm, norm(Q[:, j]) / norm(R[:, j]))
        (Q, _) = qr(Q, mode='economic')

    if n_iter > 0:
        (Q, _) = lu(Q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(n_iter):
        cnorm = np.zeros((l))
        for j in range(l):
            cnorm[j] = norm(Q[:, j])

        Q = mult(A, Q)
        if it + 1 < n_iter:
            (Q, _) = lu(Q, permute_l=True)
        else:
            anorm = 0
            for j in range(l):
                anorm = max(anorm, norm(Q[:, j]) / cnorm[j])
            (Q, _) = qr(Q, mode='economic')

    # Use the Nystrom method to obtain approximations to the
    # eigenvalues and eigenvectors of A (shifting A on the subspace
    # spanned by the columns of Q in order to make the shifted A be
    # positive definite). An alternative is to use the (symmetric)
    # square root in place of the Cholesky factor of the shift.
    anorm = .1e-6 * anorm * math.sqrt(1. * n)
    E = mult(A, Q) + anorm * Q
    R = Q.conj().T.dot(E)
    R = (R + R.conj().T) / 2
    R = cholesky(R, lower=True)
    (E, d, V) = svd(solve(R, E.conj().T), full_matrices=False)
    V = V.conj().T
    d = d * d - anorm

    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    idx = abs(d).argsort()[-k:][::-1]
    return abs(d[idx]), V[:, idx]


def eigens(A, k=6, n_iter=4, l=None):
    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Check whether A is self-adjoint to nearly the machine precision.
    x = np.random.uniform(low=-1.0, high=1.0, size=(n, 1))
    y = mult(A, x)
    z = mult(x.conj().T, A).conj().T
    assert (norm(y - z) <= .1e-11 * norm(y)) and \
           (norm(y - z) <= .1e-11 * norm(z))

    # Eigendecompose A directly if l >= n/1.25.
    if l >= n / 1.25:
        (d, V) = eigh(A.todense() if issparse(A) else A)
        # Retain only the entries of d with the k greatest absolute
        # values and the corresponding columns of V.
        idx = abs(d).argsort()[-k:][::-1]
        return d[idx], V[:, idx]

    # Apply A to a random matrix, obtaining Q.
    if isreal:
        Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
    if not isreal:
        Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l))
                 + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)))

    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    if n_iter == 0:
        (Q, _) = qr(Q, mode='economic')
    if n_iter > 0:
        (Q, _) = lu(Q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(n_iter):
        Q = mult(A, Q)
        if it + 1 < n_iter:
            (Q, _) = lu(Q, permute_l=True)
        else:
            (Q, _) = qr(Q, mode='economic')

    # Eigendecompose Q'*A*Q to obtain approximations to the eigenvalues
    # and eigenvectors of A.
    R = Q.conj().T.dot(mult(A, Q))
    R = (R + R.conj().T) / 2
    (d, V) = eigh(R)
    V = Q.dot(V)

    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    idx = abs(d).argsort()[-k:][::-1]
    return d[idx], V[:, idx]


def pca(A, k=6, raw=False, n_iter=2, l=None):
    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert k > 0
    assert k <= min(m, n)
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    if raw:
        # SVD A directly if l >= m/1.25 or l >= n/1.25.
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd(A.todense() if issparse(A) else A, full_matrices=False)
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:
            # Apply A to a random matrix, obtaining Q.
            if isreal:
                Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
            if not isreal:
                Q = mult(A,
                         np.random.uniform(low=-1.0, high=1.0, size=(n, l)) + 1j * np.random.uniform(low=-1.0, high=1.0,
                                                                                                     size=(n, l)))

            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            # Conduct normalized power iterations.
            for it in range(n_iter):
                Q = mult(Q.conj().T, A).conj().T
                (Q, _) = lu(Q, permute_l=True)
                Q = mult(A, Q)
                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            # SVD Q'*A to obtain approximations to the singular values
            # and right singular vectors of A; adjust the left singular
            # vectors of Q'*A to approximate the left singular vectors
            # of A.
            QA = mult(Q.conj().T, A)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:
            # Apply A' to a random matrix, obtaining Q.
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m))
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    + 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m))

            Q = mult(R, A).conj().T

            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            # Conduct normalized power iterations.
            for it in range(n_iter):

                Q = mult(A, Q)
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(Q.conj().T, A).conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            # SVD A*Q to obtain approximations to the singular values
            # and left singular vectors of A; adjust the right singular
            # vectors of A*Q to approximate the right singular vectors
            # of A.
            (U, s, Ra) = svd(mult(A, Q), full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            return U[:, :k], s[:k], Va[:k, :]

    if not raw:
        # Calculate the average of the entries in every column.
        c = A.sum(axis=0) / m
        c = c.reshape((1, n))

        # SVD the centered A directly if l >= m/1.25 or l >= n/1.25.
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd((A.todense() if issparse(A) else A) - np.ones((m, 1)).dot(c), full_matrices=False)
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:
            # Apply the centered A to a random matrix, obtaining Q.
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l))
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) + 1j * np.random.uniform(low=-1.0, high=1.0,
                                                                                                size=(n, l))

            Q = mult(A, R) - np.ones((m, 1)).dot(c.dot(R))

            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            # Conduct normalized power iterations.
            for it in range(n_iter):

                Q = (mult(Q.conj().T, A) - (Q.conj().T.dot(np.ones((m, 1)))).dot(c)).conj().T
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q) - np.ones((m, 1)).dot(c.dot(Q))

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            # SVD Q' applied to the centered A to obtain
            # approximations to the singular values and right singular
            # vectors of the centered A; adjust the left singular
            # vectors to approximate the left singular vectors of the
            # centered A.
            QA = mult(Q.conj().T, A) - (Q.conj().T.dot(np.ones((m, 1)))).dot(c)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:
            # Apply the adjoint of the centered A to a random matrix,
            # obtaining Q.
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m))
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) + 1j * np.random.uniform(low=-1.0, high=1.0,
                                                                                                size=(l, m))

            Q = (mult(R, A) - (R.dot(np.ones((m, 1)))).dot(c)).conj().T

            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            # Conduct normalized power iterations.
            for it in range(n_iter):
                Q = mult(A, Q) - np.ones((m, 1)).dot(c.dot(Q))
                (Q, _) = lu(Q, permute_l=True)
                Q = (mult(Q.conj().T, A) - (Q.conj().T.dot(np.ones((m, 1)))).dot(c)).conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            # SVD the centered A applied to Q to obtain approximations
            # to the singular values and left singular vectors of the
            # centered A; adjust the right singular vectors to
            # approximate the right singular vectors of the centered A.
            (U, s, Ra) = svd(mult(A, Q) - np.ones((m, 1)).dot(c.dot(Q)), full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            return U[:, :k], s[:k], Va[:k, :]


def mult(A, B):
    if issparse(B) and not issparse(A):
        # dense.dot(sparse) is not available in scipy.
        return B.T.dot(A.T).T
    else:
        return A.dot(B)


def set_matrix_mult(newmult):
    global mult
    mult = newmult


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()


'''
    The main purpose of PCA is the analysis of data to identify patterns and finding patterns
    to reduce the dimension of the data set with minimal loss of information.
    Namely, the desired outcome is to project a feature space(our data set consisting of n-dimensional
    samples) onto a smaller subspace that could represent our data set 'well'.
'''

# np.random.seed(2**32-432) # random seed for consistency
#
# mu_vec1 = np.array([0, 0, 0])
# cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
# assert class1_sample.shape == (3, 20), "The matrix has not the dimension 3x20."
#
# mu_vec2 = np.array([1, 1, 1])
# cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
# assert class2_sample.shape == (3, 20), "The matrix has not the dimensions 3x20."
#
# # 1. Taking the whole data set ignoring the class labels
# all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
# assert all_samples.shape == (3, 40), "The matrix has not the dimensions 3x40."
#
# # 2 Computing the d-dimensional mean vector
# mean_x = np.mean(all_samples[0, :])
# mean_y = np.mean(all_samples[1, :])
# mean_z = np.mean(all_samples[2, :])
#
# mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
# print ('Mean Vector: \n', mean_vector)
#
# # 3.a Computing the Scatter Matrix
# scatter_matrix = np.zeros((3, 3))
# for i in range(all_samples.shape[1]):
#     scatter_matrix += (all_samples[:, i].reshape(3, 1) - mean_vector).dot((all_samples[:, i].reshape(3, 1) - mean_vector).T)
# print ('Scatter Matrix: \n', scatter_matrix)
#
# # 3.b Computing the Covariance Matrix (alternatively to the scatter matrix)
# cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
# print ('Covariance Matrix: \n', cov_mat)
#
# # 4 Computing eigenvectors and corresponding eigenvalues
# # eigen vectors and eigen values for the from the scatter matrix
# eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
# # eigen vectors and eigen values for the from the covariance matrix
# eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
#
# for i in range(len(eig_val_sc)):
#     eigvec_sc = eig_vec_sc[:, i].reshape(1, 3).T
#     eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T
#     assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical.'
#
#     print ('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
#     print ('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
#     print ('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
#     print ('Scaling factor: ', eig_val_sc[i] / eig_val_cov[i])
#     print (40*'-')
#
# # Checking the eigenvector-eigenvalue calculation
# for i in range(len(eig_val_sc)):
#     eigv = eig_vec_sc[:, i].reshape(1, 3).T
#     np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i]*eigv, decimal=6, err_msg='', verbose=True)
#
# # Visualizing the eigen vectors, plot the eigen vectors centered at the sample mean
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs
#
#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)
#
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot(all_samples[0, :], all_samples[1, :], all_samples[2, :], 'o', markersize=8, color='green', alpha=0.2)
# ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
# for v in eig_vec_sc.T:
#     a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
#     ax.add_artist(a)
# ax.set_xlabel('x_values')
# ax.set_ylabel('y_values')
# ax.set_zlabel('z_values')
#
# plt.title('Eigenvectors')
#
# plt.show()
#
# # 5.1 Sorting the eigenvectors by decreasing eigenvalues
# for ev in eig_vec_sc:
#     np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
# eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
# eig_pairs.sort(key=lambda x: x[0], reverse=True)
#
# for i in eig_pairs:
#     print (i[0])
#
# # 5.2 Choosing k eigenvectors with the largest eigenvalues
# matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
# print ('Matrix W: \n', matrix_w)
#
# # 6 Transforming the samples onto the new subspace
# transformed = matrix_w.T.dot(all_samples)
# assert transformed.shape == (2, 40), 'The matrix has not the dimensions 2x40.'
#
# plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
# plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='red', alpha=0.5, label='class1')
# plt.xlim([-4, 4])
# plt.ylim([-4, 4])
# plt.xlabel('x_values')
# plt.ylabel('y_values')
# plt.legend()
# plt.title('Transformed samples with class labels')
#
# plt.show()
#
#
# class PCA():
#
#     def __init__(self): pass
#
#     def transform(self, X, n_components):
#         pass
#
#     def get_color_map(self, N):
#         pass
#
#     def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_lables=None):
#         pass
#
#     def plot_in_3d(self, X, y=None):
#         pass
#
#
#
# if __name__ == "__main__":
#     print (class2_sample)
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     plt.rcParams['legend.fontsize'] = 10
#     ax.plot(class1_sample[0, :], class1_sample[1, :], class1_sample[2, :], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
#     ax.plot(class2_sample[0, :], class2_sample[1, :], class2_sample[2, :], '^', markersize=8, color='red', alpha=0.5, label='class2')
#     plt.title('Samples for class 1 and class 2')
#     ax.legend(loc='upper right')
#
#     plt.show()



