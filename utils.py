import numpy as np
from numba import njit
from scipy import linalg, stats
from tqdm import tqdm
from tensorly.base import unfold
from tensorly.decomposition import parafac
from scipy.sparse.linalg import eigsh


def dichotomy(func, x0, step0=1, delta=1e-5):
    # Finds the root of a function using dichotomy.
    xp, xm = x0, np.nan
    step = step0
    s = 1 if func(xp) < 0 else -1
    while step > delta:
        xm = xp
        xp = xm+s*step
        while np.isnan(func(xp)): # avoid nan
            step /= 2
            xp = xm+s*step
        if func(xm)*func(xp) <= 0:
            step /= 2
            s *= -1
    xp = xp+s*step
    return xp

# TENSOR MODEL

# Stieltjes transform
@njit
def g(ksi, c, beta_T, delta=1e-5, max_ite=10000):
    ggp, ggm = np.ones(3, dtype='c16')*1j, np.zeros(3, dtype='c16')
    N_ite = 0
    # Fixed point
    while (np.abs(np.sum(ggp)-np.sum(ggm)) > delta) and (N_ite < max_ite):
        gamma = beta_T**2*(1-ggp[2]**2/c[2])/(c[0]+c[1])
        ggm[0], ggm[1], ggm[2] = ggp[0], ggp[1], ggp[2]
        ggp[0] = c[0]/(-ggp[1]-ggp[2]-gamma*ggp[1]-ksi)
        ggp[1] = c[1]/(-ggp[0]-ggp[2]-gamma*ggp[0]-ksi)
        ggp[2] = c[2]/(-ggp[0]-ggp[1]-ksi)
        N_ite += 1
    return ggp

# Summary statistics
def align(c, beta_T, beta_M, delta=1e-5, eps=1e-3):
    
    def f(ksi):
        gg = g(ksi, c, beta_T, delta)
        if np.sum(gg).imag > eps:
            return gg, np.nan, np.zeros(3)*np.nan
        gg = gg.real
        qq2 = np.zeros(3)
        qq2[2] = 1-gg[2]**2/c[2]
        gamma = beta_T**2*qq2[2]/(c[0]+c[1])
        qq2[:2] = 1-(1+gamma)*gg[:2]**2/c[:2]
        return gg, gamma, qq2
    
    def func(ksi):
        gg, gamma, qq2 = f(ksi)
        if np.isnan(gamma) or np.any(qq2 < 0):
            return np.nan
        return ksi+(1+gamma)*np.sum(gg)-gamma*gg[2]-beta_T*beta_M*np.sqrt(np.prod(qq2))
    
    ksi0 = 1
    while np.abs(np.sum(g(ksi0, c, beta_T)).imag) > eps:
        ksi0 += 1 # go far enough of the bulk
    lbda = dichotomy(func, ksi0, delta=delta) # solve func = 0
    qq2 = f(lbda)[2]
    if np.any(qq2 < 0):
        qq2 *= 0
    return lbda, np.sqrt(qq2)


# MULTI-VIEW CLUSTERING

# Model generating function
def model(p, n, m, mu_norm, h_norm, y):
    mu, h = np.random.randn(p), np.random.randn(m)
    mu = mu_norm*mu/linalg.norm(mu)
    h = h_norm*h/linalg.norm(h)
    Z = np.random.randn(p, n)/np.sqrt(p+n)
    W = np.random.randn(p, n, m)/np.sqrt(p+n+m)
    return np.einsum('ij,k->ijk', np.einsum('i,j->ij', mu, y/np.sqrt(n))+Z, h)+W

def unfolding_clustering(X):
    X2 = unfold(X, 1)
    return eigsh(X2@X2.T, k=1)[1][:, 0]

def tensor_clustering(X):
    return parafac(X, 1, normalize_factors=True)[1][1][:, 0]

def unfolding_accuracy(p, n, m, mu_norm, h_norm):
    s = p+n+m
    rho, mu2 = np.meshgrid(h_norm**2*s/np.sqrt(p*n*m), mu_norm**2)
    cp, cn, cm = p/s, n/s, m/s
    with np.errstate(divide='ignore'):
        zeta = 1-((mu2/(rho*(cn/(1-cm)+mu2)))**2+cn*(cp/(1-cm)+mu2)/(1-cm))/(mu2*(cn/(1-cm)+mu2))
    zetap = np.maximum(zeta, 0)
    return stats.norm.cdf(np.sqrt(zetap/(1-zetap)))

def tensor_accuracy(p, n, m, mu_norm, h_norm):
    s = p+n+m
    c = np.array([p/s, n/s, m/s])
    acc = np.zeros((len(mu_norm), len(h_norm)))
    for i, beta_M in enumerate(tqdm(mu_norm)):
        for j, beta_T in enumerate(h_norm):
            alpha = align(c, beta_T, beta_M)[1][1]
            acc[i, j] = stats.norm.cdf(alpha/np.sqrt(1-alpha**2))
    return acc
