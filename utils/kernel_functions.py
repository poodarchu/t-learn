from __future__ import division
import numpy as np

# The Linear kernel is the simplest kernel function.
# It is given by the inner product <x,y> plus an optional constant c.
# Kernel algorithms using a linear kernel are often equivalent to their non-kernel counterparts,
# i.e. KPCAwith linear kernel is the same as standard PCA.
def linear_kernel(constant=0, **kwargs):
    def f(x, y):
        return np.inner(x, y) + constant
    return f

# The Polynomial kernel is a non-stationary kernel.
# Polynomial kernels are well suited for problems where all the training data is normalized.
def polynomial_kernel(power=2, constant=0, **kwargs):
    def f(x, y):
        return (np.inner(x, y) + constant)**power
    return f

# The Gaussian kernel is an example of radial basis function kernel
def gaussian_kernel(sigma=1, **kwargs):
    def f(x, y):
        return np.exp(- np.linalg.norm(x-y)**2 / (2*sigma**2))
    return f

# The exponential kernel is closely related to the Gaussian kernel,
# with only the square of the norm left out. It is also a radial basis function kernel.
def exponential_kernel(sigma=1, **kwargs):
    def f(x, y):
        return np.exp(- np.linalg.norm(x-y) / 2*sigma**2)
    return f

# The Laplace Kernel is completely equivalent to the exponential kernel,
# except for being less sensitive for changes in the sigma parameter.
# Being equivalent, it is also a radial basis function kernel.
def laplacian_kernel(sigma=1, **kwargs):
    def f(x, y):
        return np.exp(- np.linalg.norm(x-y) / sigma)
    return f

def rbf_kernel(gamma=1, **kwargs):
    def f(x, y):
        return np.exp(- gamma * np.linalg.norm(x-y)**2)
    return f






