"""
Methods for approximating entropy of Gaussian mixtures. Implementation of
Huber et al. 'On Entropy Approximation for Gaussian Mixture Random Vectors'
(2008)
"""
import numpy as np
from scipy.misc import logsumexp
from scipy.linalg import solve, solve_triangular

def H0(w, mu, Lambda, chol=False):
    """
    Calculate entropy based on 0th-order Taylor expansion of the logarithm
    about each of the mixture component means.

    Parameters:
        w: (K,) numpy vector of weights.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
    """
    return -w.dot(mv_logpdf(mu, w, mu, Lambda, chol))

def mv_logpdf_components(x, mu, Lambda, chol=False):
    """
    Log probability density function for each component of mixture of K
    multivariate normals with mean mu and precision Lambda.

    Parameters:
        x: (N, D) numpy array of N D-vector inputs.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
        chol: is Lambda in fact the Cholesky factor L of the precision?

    Returns:
        lpdf: (N, K) numpy array of logpdfs for each input, mixture component
    """
    dx = x[:, np.newaxis] - mu[np.newaxis]
    if chol:
        vv = np.einsum('kji, nkj -> nki', Lambda, dx)
        lpdf = np.einsum('nkj, nkj -> nk', vv, vv)
        lpdf *= -0.5
        lpdf += -0.5 * D * np.log(2 * np.pi)
        Ltrace = np.diagonal(L, axis1=1, axis2=2)
        lpdf += np.sum(np.log(Ltrace), axis=1)
    else:
        lpdf = np.einsum('nkj, kji, nki -> nk', dx, Lambda, dx)
        lpdf *= -0.5
        lpdf += -0.5 * D * np.log(2 * np.pi)
        lpdf += 0.5 * np.linalg.slogdet(Lambda)[1]

    return lpdf


def mv_logpdf(x, w, mu, Lambda, chol=False):
    """
    Log probability density function of mixture of K multivariate normals with mean mu and precision Lambda.

    Parameters:
        x: (N, D) numpy array of N D-vector inputs.
        w: (K,) numpy vector of weights.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
        chol: is Lambda in fact the Cholesky factor L of the precision?

    Returns:
        (N,) numpy array of logpdfs for each input
    """
    return logsumexp(np.log(w) + mv_logpdf_components(x, mu, Lambda, chol), axis=1)

def _calc_rel_pdf_components(x, mu, Lambda, chol):
    logg = mv_logpdf_components(x, mu, Lambda, chol)
    # now normalize relative to smallest probability to prevent underflow
    logg = logg - np.min(logg, axis=1, keepdims=True)

    return w * np.exp(logg)

def _calc_vector_components(x, mu, Lambda, chol):
    dx = x[:, np.newaxis] - mu[np.newaxis]
    if chol:
        m = np.einsum('kij, klj, nkl -> nki', Lambda, Lambda, dx)
    else:
        m = np.einsum('kij, nkj -> nki', Lambda, dx)

    return m

def normed_grad(x, w, mu, Lambda, chol=False):
    """
    Gradient of the Gaussian mixture model divided by pdf (d(log g) = dg / g)

    Parameters:
        x: (N, D) numpy array of N D-vector inputs.
        w: (K,) numpy vector of weights.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
        chol: is Lambda in fact the Cholesky factor L of the precision?

    Returns:
        (N, D) numpy array of D-vector normed gradients (one per example)
    """
    wg = _calc_rel_pdf_components(x, mu, Lambda, chol)
    m = _calc_vector_components(x, mu, Lambda, chol)

    return np.einsum('nk, nki -> ni', wg, m)/np.sum(wg, axis=1, keepdims=True)

def normed_hess(x, w, mu, Lambda, chol=False):
    """
    Hessian of the Gaussian mixture model divided by pdf (Hg / g)

    Parameters:
        x: (N, D) numpy array of N D-vector inputs.
        w: (K,) numpy vector of weights.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
        chol: is Lambda in fact the Cholesky factor L of the precision?

    Returns:
        (N, D, D) numpy array of D-matrix normed Hessians (one per example)
    """
    wg = _calc_rel_pdf_components(x, mu, Lambda, chol)
    m = _calc_vector_components(x, mu, Lambda, chol)

    if chol:
        Prec = np.einsum('nij, nlj -> nil', Lambda, Lambda)
    else:
        Prec = Lambda
    H = np.einsum('nki, nkj -> nkij', m, m) - Prec

    sum_wg = np.sum(wg, axis=1)

    return np.einsum('nk, nkij -> nij', wg, H)/sum_wg[:, np.newaxis, np.newaxis]

def H2(w, mu, Lambda, chol=False):
    """
    Calculate entropy based on 0th-order Taylor expansion of the logarithm
    about each of the mixture component means.

    Parameters:
        w: (K,) numpy vector of weights.
        mu: (K, D) numpy array of means.
        Lambda: (K, D, D) numpy array of precision matrices.
    """
    K = w.shape[0]
    h = normed_grad(mu, w, mu, Lambda, chol)
    H = normed_hess(mu, w, mu, Lambda, chol)

    HH = 0
    if chol:
        Prec = np.einsum('nij, nlj -> nil', Lambda, Lambda)
        for idx in range(K):
            Linvh = solve_triangular(Lambda[idx], h[idx], lower=True)
            LinvH = solve_triangular(Lambda[idx], H[idx], lower=True)
            LaminvH = solve_triangular(Lambda[idx].T, LinvH, lower=False)
            HH += w[idx] * (-Linvh.dot(Linvh) + np.trace(LaminvH))
    else:
        Prec = Lambda
        for idx in range(K):
            Laminvh = solve(Lambda[idx], h[idx])
            LaminvH = solve(Lambda[idx], H[idx])
            HH += w[idx] * (-h[idx].dot(Laminvh) + np.trace(LaminvH))

    return -0.5 * HH

if __name__ == '__main__':
    import scipy.stats as stats
    import numpy.testing as npt

    np.random.seed(12345)
    K, D, N = 3, 5, 7
    mu = np.random.randn(K, D)
    x = mu[0] + 0.01 * np.random.randn(N, D)
    L = 5 * np.tril(np.random.randn(K, D, D))
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

    # test normed_grad, normed_hess
    log_comps = mv_logpdf_components(x, mu, Lambda)
    wg = w * np.exp(log_comps)
    xx = x
    mlist = []
    Hlist = []
    for idx in range(K):
        mm = mu[idx]
        dx = xx - mm
        this_m = Lambda[idx].dot(dx.T).T
        mlist.append(this_m)

        this_H = np.einsum('ni, nj -> nij', this_m, this_m) - Lambda[idx]
        Hlist.append(this_H)

    m = np.array(mlist)
    Hg = np.array(Hlist)
    wg_sum = np.sum(wg, axis=1)
    normed_g = np.einsum('nk, kni -> ni', wg, m)/wg_sum[:, np.newaxis]
    npt.assert_allclose(normed_g, normed_grad(x, w, mu, Lambda))
    npt.assert_allclose(normed_g, normed_grad(x, w, mu, L, True))

    normed_H = np.einsum('nk, knij -> nij', wg, Hg)/wg_sum[:, np.newaxis, np.newaxis]
    npt.assert_allclose(normed_H, normed_hess(x, w, mu, Lambda))
    npt.assert_allclose(normed_H, normed_hess(x, w, mu, L, True))

    # test H2
    hh = normed_grad(mu, w, mu, Lambda)
    HH = normed_hess(mu, w, mu, Lambda)

    Laminv = np.array([np.linalg.inv(Lambda[k]) for k in range(K)])
    terms = [-hh[k].dot(Laminv[k].dot(hh[k])) + np.trace(Laminv[k].dot(HH[k])) for k in range(K)]
    np_H2 = -0.5 * np.sum(w * np.array(terms))
    this_H2 = H2(w, mu, Lambda)
    npt.assert_allclose(np_H2, H2(w, mu, Lambda))
    npt.assert_allclose(np_H2, H2(w, mu, L, True))
