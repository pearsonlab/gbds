"""
Methods for approximating entropy of Gaussian mixtures. Implementation of
Huber et al. 'On Entropy Approximation for Gaussian Mixture Random Vectors'
(2008)
"""
import numpy as np
from scipy.misc import logsumexp

def H0(w, mu, Lambda, chol=False):
    """
    Calculate entropy based on 0th-order Taylor expansion of the logarithm
    about each of the mixture component means.

    Parameters:
        w: (N,) numpy vector of weights.
        mu: (N, D) numpy array of means.
        Lambda: (N, D, D) numpy array of precision matrices.
    """
    return -w.dot(mv_logpdf(mu, w, mu, Lambda, chol))

def mv_logpdf(x, w, mu, Lambda, chol=False):
    """
    Log probability density function of mixture of K multivariate normals with mean mu and precision Lambda.

    Parameters:
        x: (N, D) numpy array of N D-vector inputs.
        w: (K,) numpy vector of weights.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
        chol: is Lambda in fact the Cholesky factor L of the precision?
    """
    if chol:
        dx = x[:, np.newaxis] - mu[np.newaxis]
        vv = np.einsum('kji, nkj -> nki', Lambda, dx)
        lpdf = np.einsum('nkj, nkj -> nk', vv, vv)
        lpdf *= -0.5
        lpdf += -0.5 * D * np.log(2 * np.pi)
        Ltrace = np.diagonal(L, axis1=1, axis2=2)
        lpdf += np.sum(np.log(Ltrace), axis=1)
    else:
        dx = x[:, np.newaxis] - mu[np.newaxis]
        lpdf = np.einsum('nkj, kji, nki -> nk', dx, Lambda, dx)
        lpdf *= -0.5
        lpdf += -0.5 * D * np.log(2 * np.pi)
        lpdf += 0.5 * np.linalg.slogdet(Lambda)[1]

    return logsumexp(np.log(w) + lpdf, axis=1)

if __name__ == '__main__':
    import scipy.stats as stats
    import numpy.testing as npt

    np.random.seed(12345)
    K, D, N = 3, 5, 7
    mu = np.random.randn(K, D)
    x = mu[0] + 0.01 * np.random.randn(N, D)
    L = np.tril(np.random.randn(K, D, D))
    for k in range(K):
        for d in range(D):
            L[k, d, d] = np.abs(L[k, d, d])
    Lambda = np.einsum('ijk, ilk -> ijl', L, L)
    w = np.random.rand(K)
    w /= np.sum(w)

    # test mv_logpdf
    lpdfs = []
    xx = x
    for idx in range(K):
        mm = mu[idx]
        Sig = np.linalg.inv(Lambda[idx])
        lpdfs.append(stats.multivariate_normal.logpdf(xx, mean=mm, cov=Sig))

    np_lpdf = logsumexp(np.log(w) + np.array(lpdfs).T, axis=1)

    this_lpdf = mv_logpdf(x, w, mu, Lambda)
    this_lpdf_chol = mv_logpdf(x, w, mu, L, True)

    npt.assert_allclose(np_lpdf, this_lpdf)
    npt.assert_allclose(np_lpdf, this_lpdf_chol)

    # test H0
    lpdfs = []
    xx = mu
    for idx in range(K):
        mm = mu[idx]
        Sig = np.linalg.inv(Lambda[idx])
        lpdfs.append(stats.multivariate_normal.logpdf(xx, mean=mm, cov=Sig))

    np_lpdf = logsumexp(np.log(w) + np.array(lpdfs).T, axis=1)
    np_H0 = -w.dot(np_lpdf)

    this_H0 = H0(w, mu, Lambda)
    this_H0_chol = H0(w, mu, L, True)

    npt.assert_allclose(np_H0, this_H0)
    npt.assert_allclose(np_H0, this_H0_chol)
