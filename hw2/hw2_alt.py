__author__ = 'Bob Forcha'
"""
A script to numerically approximate the Poisson equation under a number of scenarios.

MAE 6225 - CFD
The George Washington University
Spring 2017
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# construct mesh grid for domain and applying BCs
ni, nj = 7, 7
n = range(1, int((np.minimum(ni, nj)-1)/2))
dx, dy = 1.0/(ni-1), 1.0/(nj-1)
x_start, x_end = -dx/2, 1 + dx/2
y_start, y_end = -dy/2, 1 + dy/2
x = np.linspace(x_start, x_end, ni+1)
y = np.linspace(y_start, y_end, nj+1)
X, Y = np.meshgrid(x, y)
ij = len(x)*len(y)

# givens
u_ex_tmp = np.sin(2*math.pi*n[0]*X)*np.sin(2*math.pi*n[0]*Y)
u_ex = np.reshape(u_ex_tmp, (ij, 1))

# create linear system Au=f to represent approximation
u0 = np.zeros(np.shape(u_ex), dtype=float)
A = -4.0*np.eye(ij) + np.eye(ij, k=1) + np.eye(ij, k=-1)
for num in range(2, int(len(u0)/2) + 1):
    if num % ni == 0:
        A += np.eye(ij, k=num)
        A += np.eye(ij, k=-num)

f = -8*math.pi**2*n[0]**2*u_ex*dx*dy


def jacobi(A, f, n_max, u0):
    """
    Solves Au=f.  Returns u.

    :param A:
    :param f:
    :param n_max:
    :param u0:
    :return u:
    """

    d = -dx*dy/4
    d_inv = 1/d
    b = d_inv*f
    err = 1
    n_count = 0
    u = u0

    while n_count < n_max:
        u_old = np.reshape(u, np.shape(X))
        for j in range(1, len(u_old[:, 0])-1):
            for i in range(1, len(u_old[0, :])-1):
                u