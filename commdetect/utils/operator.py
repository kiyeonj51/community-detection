import numpy as np


def proj_simplex(v, s=1):
    n = v.shape[0]
    if v.sum() == s and np.alltrue(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = float(cssv[rho] - s) / rho
    w = (v - theta).clip(min=0)
    return w


def proj_l1(v, s=1):
    n = v.shape[0]
    u = np.abs(v)
    if u.sum() <= s:
        return v
    w = proj_simplex(u, s=s)
    w *= np.sign(v)
    return w


def proj_1inf(Y, lamb):
    res = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        res[i, :] = Y[i, :] - proj_l1(Y[i, :], lamb)
    return res


def proj_A(X):
    n = X.shape[0]
    temp1 = np.append((X - np.diag(np.diag(X))).dot(np.ones(n)) * 2, np.diag(X)) - b
    mu1, nu1 = temp1[:n], temp1[n:]
    temp2 = np.append(1 / (2 * (n - 2)) * (mu1 - np.ones(n) * (np.sum(mu1) / (2 * n - 2))), nu1)
    mu2, nu2 = temp2[:n], temp2[n:]
    temp = np.outer(mu2, np.ones(n))
    res = X - ((temp + temp.T - 2 * np.diag(mu2)) + np.diag(nu2))
    return res


def proj_psd(X):
    w, v = np.linalg.eig((X + X.T) / 2)
    idx = (w >= 0)
    res = v[:, idx].dot(np.diag(w[idx])).dot(v[:, idx].T)
    return res


def softth(X, lamb):
    return np.sign(X) * np.maximum(np.abs(X) - lamb, 0)

def proj_mani(sigma):
    res=sigma/np.outer(np.sqrt(np.sum(sigma**2,axis=1)),np.ones(sigma.shape[1]))
    return res


