from __future__ import division
import numpy as np
from commdetect.core.base import SDP
from commdetect.core.base import BurerMonteiro
import matplotlib.pylab as plt
from commdetect.utils.operator import (proj_1inf)
import time

# print(__doc__)

# Example setting
n = 100
psi = np.asarray([
    [.300, .200, .150],
    [.200, .300, .100],
    [.150, .100, .300]
])

# Define network
Z1_gt = np.asarray([1, 0, 0] * (2 * n) + [0, 1, 0] * n + [0, 0, 1] * n).reshape(4 * n, 3)
Mz1 = Z1_gt.dot(psi).dot(Z1_gt.T)
X1_gt = Z1_gt.dot(Z1_gt.T)

Z2_gt = np.asarray([1, 0, 0] * n + [0, 1, 0] * (2 * n) + [0, 0, 1] * n).reshape(4 * n, 3)
Mz2 = Z2_gt.dot(psi).dot(Z2_gt.T)
X2_gt = Z2_gt.dot(Z2_gt.T)

Z3_gt = np.asarray([1, 0, 0] * n + [0, 1, 0] * n + [0, 0, 1] * (2 * n)).reshape(4 * n, 3)
Mz3 = Z3_gt.dot(psi).dot(Z3_gt.T)
X3_gt = Z3_gt.dot(Z3_gt.T)

# Define plots

# Generate dataset
A1 = np.random.binomial(1, Mz1)
A1 = np.triu(A1) + np.triu(A1).T - 2 * np.diag(np.diag(A1))

A2 = np.random.binomial(1, Mz2)
A2 = np.triu(A2) + np.triu(A2).T - 2 * np.diag(np.diag(A2))

A3 = np.random.binomial(1, Mz3)
A3 = np.triu(A3) + np.triu(A3).T - 2 * np.diag(np.diag(A3))

# Define algorithms
commdetect_algorithms = [
    ("SDP", SDP()),
    ("Burer-Monteiro", BurerMonteiro(k=5, eta=5.))
]

datasets = [(A1, X1_gt), (A2, X2_gt), (A3, X3_gt)]

# Reults
result = {"SDP": [], "Burer-Monteiro": []}

# Compare algorithms
for i_dataset, (A, X_gt) in enumerate(datasets):
    print(f"{i_dataset} dataset")
    for name, algorithm in commdetect_algorithms:
        print(f"{name}")
        res = algorithm.solve(A=A, X_gt=X_gt)
        result[name].append(res)
        plt.figure()
        plt.imshow(res)
        plt.colorbar()
        plt.show()

for name, _ in commdetect_algorithms:
    res = result[name]
    C_opt = np.vstack((np.vstack((np.ravel(res[0]), np.ravel(res[1]))), np.ravel(res[2]))).T
    C_opt = proj_1inf(C_opt, 1.1).T
    C1_opt = C_opt[0, :].reshape(4 * n, 4 * n)
    C2_opt = C_opt[1, :].reshape(4 * n, 4 * n)
    C3_opt = C_opt[2, :].reshape(4 * n, 4 * n)

    plt.figure(3)
    plt.subplot(131)
    plt.title('C1_opt')
    plt.imshow(C1_opt)

    plt.subplot(132)
    plt.title('C2_opt')
    plt.imshow(C2_opt)

    plt.subplot(133)
    plt.title('C3_opt')
    plt.imshow(C3_opt)
    plt.show()
