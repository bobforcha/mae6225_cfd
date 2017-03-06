__author__ = 'Bob Forcha'

"""
this script numerically approximates an exact solution of the Poisson equation using the jacobi method
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# create mesh grid to represent domain in x and y
ni, nj = 9, 9
n = range(1, int((np.minimum(ni, nj)-1)/2))
dx, dy = 1/(ni-1), 1/(nj-1)
x_start, x_end = -dx/2, 1 + dx/2
y_start, y_end = -dy/2, 1 + dy/2
x = np.linspace(x_start, x_end, ni)
y = np.linspace(y_start, y_end, nj)
X, Y = np.meshgrid(x, y)

# exact solution
u_ex = np.zeros((len(n), np.shape(X)[0], np.shape(X)[1]))
rhs = np.zeros(np.shape(u_ex))
for num in range(0, len(n)):
    u_ex[num, :, :] = np.cos(2*math.pi*n[num]*X)*np.cos(2*math.pi*n[num]*Y)
    rhs[num, :, :] = -8*math.pi**2*n[num]**2*u_ex[num, :, :]


# define jacobi function
def jacobi(phi_0, f, dx, dy, err_min, n, n_max, phi_ex):
    err = 1
    phi = phi_0
    n_cnt = 0

    while err > err_min and n_cnt < n_max:
        phi_old = phi
        for j in range(1, nj-1):
            for i in range(1, ni-1):
                phi[j, i] = 1/4*(phi_old[j, i-1] + phi_old[j, i+1] + phi_old[j-1, i] + phi_old[j+1, i])\
                            - (dx*dy)/4*f[j, i]

        phi[0, 0] = 1
        phi[1, :] = phi[0, :] - 2*math.pi*n*dy
        phi[-2, :] = phi[-1, :] - 2*math.pi*n*dy*phi[-1, :]
        phi[:, 1] = phi[:, 0] - 2*math.pi*n*dx*phi[:, 0]
        phi[:, -2] = phi[:, -1] - 2*math.pi*n*dx*phi[:, -1]

        err = np.max(np.abs(phi - phi_ex))
        n_cnt += 1

    return phi

# approximate solution
u0 = np.ones(np.shape(u_ex))
u = u0
size = 5

for index in range(len(n)):
    u[index, :, :] = jacobi(u[index, :, :], rhs[index, :, :], dx, dy, 1e-4, n[index], 1e4, u_ex[index, :, :])

    # plot exact solution
    fig = plt.figure(figsize=(size, size))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, u_ex[index, :, :], color='g', alpha=0.8, label='Exact')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('u', fontsize=16)
    ax.set_title('Exact Solution, n = %g' % n[index], fontsize=20)

    # plot approximate solution
    fig = plt.figure(figsize=(size, size))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, u[index, :, :], color='r', alpha=0.8, label='Jacobi')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('u', fontsize=16)
    ax.set_title('Jacobi Method, n = %g' % n[index], fontsize=20)

plt.show();
