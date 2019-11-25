import numpy as np
from commdetect.utils.operator import (proj_psd, proj_mani)
import time


class SDP:
    def __init__(self, max_iter=100, rho=.05, tau=1.5, tol=1e-3):
        self.max_iter = max_iter
        self.rho = rho
        self.tau = tau
        self.tol = tol

    def init_zero(self, A):
        n = A.shape[0]
        X = np.zeros(A.shape)
        S = np.zeros(A.shape)
        Z = np.zeros(A.shape)
        v = np.zeros(n)
        y = np.zeros(n)
        s = np.zeros(n)
        return X, S, Z, v, y, s

    def solve(self, A, X_gt):
        max_iter = self.max_iter
        rho = self.rho
        tau = self.tau
        tol = self.tol
        n = A.shape[0]
        lamb = 2 / (A.shape[0] * (A.shape[0] - 1)) * np.sum(np.triu(A) - np.diag(np.diag(A)))
        b = np.ones(n)
        C = -(A - lamb * np.ones((n, n)))
        X, S, Z, v, y, s = self.init_zero(A)
        start = time.time()
        for it in range(max_iter):
            X_old = X
            R1 = np.diag(y) + S + Z - C + X / rho
            R2 = v - y + s / rho
            Z = np.minimum(np.maximum(0, rho * (R1 - Z)) / rho - (R1 - Z), 1)
            v = np.minimum(b, rho * (R2 - v)) / rho - (R2 - v)
            y = 0.5 * (v + s / rho - np.diag(S + Z - C + X / rho))
            R1 = np.diag(y) + S + Z - C + X / rho
            S = proj_psd(S - R1)
            X = X + tau * rho * (np.diag(y) + S + Z - C)
            s = s + tau * rho * (v - y)
            end = time.time()
            if it % int(max_iter/10) == 0:
                print(f'{it}-iter| X_error :{np.linalg.norm(X - X_gt)}, time : {end-start}')
            if np.linalg.norm(X - X_old) < tol:
                break
        return X


class BurerMonteiro:
    def __init__(self, max_iter=100, eta=5., k=5):
        self.max_iter = max_iter
        self.eta = eta
        self.k = k

    def solve(self, A, X_gt):
        max_iter = self.max_iter
        eta = self.eta
        k = self.k
        n = A.shape[0]
        lamb = 2 / (n * (n - 1)) * np.sum(np.triu(A) - np.diag(np.diag(A)))
        B = A - lamb * np.ones((n, n))
        sigma = proj_mani(np.random.rand(n, k))
        start = time.time()
        for it in range(max_iter):
            grad = self.gradf(B, sigma, 0)
            grad_size = np.linalg.norm(grad)

            # Grad-step
            u = grad / grad_size

            # update sigma
            sigma = sigma + eta * u
            sigma = np.maximum(sigma, 0)
            sigma = proj_mani(sigma)
            end = time.time()
            if it % int(max_iter/10) == 0:
                print(f'{it}-iter| X_error :{np.linalg.norm(sigma.dot(sigma.T) - X_gt)}, time : {end-start}')
        return sigma.dot(sigma.T)

    def gradf(self, B, sigma, reg):
        n = sigma.shape[0]
        return 2 * ((B + reg * np.eye(n)) - np.diag(np.diag((B + reg * (np.eye(n))).dot(sigma).dot(sigma.T)))).dot(sigma)