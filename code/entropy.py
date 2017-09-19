"""
Methods for approximating entropy of Gaussian mixtures. Implementation of
Huber et al. 'On Entropy Approximation for Gaussian Mixture Random Vectors'
(2008)
"""
import numpy as np
from scipy.misc import logsumexp

def H0(w, mu, Lambda):
    """
    Calculate entropy based on 0th-order Taylor expansion of the logarithm
    about each of the mixture component means.

    Parameters:
        w: (N,) numpy vector of weights.
        mu: (N, D) numpy array of means.
        Lambda: (N, D, D) numpy array of precision matrices.
    """
    pass

def mv_logpdf(x, w, mu, Lambda, chol=False):
    """
    Log probability density function of mixture of multivariate normals with mean mu and precision Lambda.

    Parameters:
        x: (D,) numpy vector of inputs.
        w: (N,) numpy vector of weights.
        mu: (N, D) numpy array of means.
        Lambda: (N, D, D) numpy array of precision matrices.
        chol: is Lambda in fact the Cholesky factor L of the precision?
    """
    if chol:
        dx = x - mu
        vv = np.einsum('ijk, ij -> ik', Lambda, dx)
        lpdf = np.einsum('ij, ij -> i', vv, vv)
        lpdf *= -0.5
        lpdf += -0.5 * D * np.log(2 * np.pi)
        Ltrace = np.diagonal(L, axis1=1, axis2=2)
        lpdf += np.sum(np.log(Ltrace), axis=1)
    else:
        dx = x - mu
        lpdf = np.einsum('ij, ijk, ik -> i', dx, Lambda, dx)
        lpdf *= -0.5
        lpdf += -0.5 * D * np.log(2 * np.pi)
        lpdf += 0.5 * np.linalg.slogdet(Lambda)[1]

    return logsumexp(np.log(w) + lpdf)

if __name__ == '__main__':
    import scipy.stats as stats
    import numpy.testing as npt

    np.random.seed(12345)
    N, D = 3, 5
    mu = np.random.randn(N, D)
    x = mu[0] + 0.01 * np.random.randn(D)
    L = np.tril(np.random.randn(N, D, D))
    for n in range(N):
        for d in range(D):
            L[n, d, d] = np.abs(L[n, d, d])
    Lambda = np.einsum('ijk, ilk -> ijl', L, L)
    w = np.random.rand(N)
    w /= np.sum(w)

    lpdfs = []
    xx = x
    for idx in range(N):
        mm = mu[idx]
        Sig = np.linalg.inv(Lambda[idx])
        lpdfs.append(stats.multivariate_normal.logpdf(xx, mean=mm, cov=Sig))

    np_lpdf = logsumexp(np.log(w) + np.array(lpdfs))

    this_lpdf = mv_logpdf(x, w, mu, Lambda)
    this_lpdf_chol = mv_logpdf(x, w, mu, L, True)

    npt.assert_approx_equal(np_lpdf, this_lpdf)
    npt.assert_approx_equal(np_lpdf, this_lpdf_chol)
