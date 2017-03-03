__author__ = 'Bob Forcha'
"""
hw2.py - a script to complete homework 2 for MAE 6225 - Computational Fluid Dynamics at
The George Washington University.

Discretization of the Poisson equation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create mesh grid
ni, nj = 11, 11
h = 1.0/(ni-1)
x_start, x_end = -h/2.0, 1.0 + h/2.0
y_start, y_end = -h/2.0, 1.0 + h/2.0
x = np.linspace(x_start, x_end, ni+1)
y = np.linspace(y_start, y_end, nj+1)
X, Y = np.meshgrid(x, y)
print(X)

# plot mesh grid to illustrate evaluation grid with ghost points
size = 5
plt.figure(figsize=(size, size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(-h, 1 + h)
plt.ylim(-h, 1 + h)
plt.grid(True)
plt.title('Grid Points', fontsize=20, fontweight='bold')
plt.scatter(X[1:nj, 1:ni], Y[1:nj, 1:ni], s=20, color='#FF0D0D', marker='o', )
plt.scatter(X[:, 0], Y[:, 0], s=20, color='k', marker='o', label='Ghost points')
plt.scatter(X[:, ni], Y[:, ni], s=20, color='k', marker='o')
plt.scatter(X[0, 1:ni], Y[0, 1:ni], s=20, color='k', marker='o')
plt.scatter(X[nj, 1:ni], Y[nj, 1:ni], s=20, color='k', marker='o')
plt.legend(loc='best')

# create linear system Au = f for numerical analysis
u0 = np.zeros(np.shape(np.reshape(X, (len(X[:, 0])*len(X[0, :]), 1))), dtype=float)
A = np.zeros((len(u0), len(u0)), dtype=float)

# populate A
A += -4*np.eye(np.shape(A)[0])
A += np.eye(np.shape(A)[0], k=1)
A += np.eye(np.shape(A)[0], k=-1)

for i in np.arange(2, np.shape(A)[0]):
    if i % ni == 0:
        A += np.eye(np.shape(A)[0], k=i)
        A += np.eye(np.shape(A)[0], k=-i)


# define Jacobi Method function
def jacobi(A, f, n_max, z):
    """
    function to solve the equation Az=f using the Jacobi iterative method
    :param A: 2D numpy array
    :param f: 1D numpy array
    :param n_max: int max number of iterations
    :param z: 1D numpy array -- initial guess for x
    :return z: 1D numpy array
    """
    D_inv = np.eye(len(A[0]))*(-h**2)/4
    DIA = np.dot(D_inv, A)
    print(DIA)
    b = np.dot(D_inv, f)

    err = 1
    while err/np.max(u_ex) > 0.001:
        for num in range(n_max):
            z_old = z
            z = np.dot((np.eye(len(z)) - DIA), z_old) + b

            # set Dirichlet BCs
            z_tmp = np.reshape(z, np.shape(X))
            z_tmp[0, 1:ni] = -z_tmp[1, 1:ni]
            z_tmp[1:nj, 0] = -z_tmp[1:nj, 1]
            z_tmp[nj, 1:ni] = -z_tmp[nj-1, 1:ni]
            z_tmp[1:nj, ni] = -z_tmp[1:nj, ni-1]
            z_tmp[0, 0] = -z_tmp[1, 1]
            z_tmp[nj, ni] = -z_tmp[nj-1, ni-1]
            z_tmp[nj, 0] = -z_tmp[nj-1, 1]
            z_tmp[0, ni] = -z_tmp[1, ni-1]
            z = np.reshape(z_tmp, (np.size(z_tmp), 1))

            # compute error
            err = np.max(z - np.reshape(u_ex, (np.shape(z))))

        return z, num

# Use Jacobi Method for Question 2
for n in range(1, int((ni)/2)):
    u_ex = np.sin(2 * math.pi * n * X) * np.sin(2 * math.pi * n * Y)
    rhs = np.reshape(-8*(math.pi**2)*(n**2)*u_ex, (np.size(u_ex), 1))
    u_comp, N = jacobi(A, rhs, 100, u0)

print(N)
# plot final values
fig = plt.figure(figsize=(size, size))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, np.reshape(u_comp, np.shape(X)), cmap=plt.cm.jet)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('u', fontsize=16)
ax.set_title('u computed with Jacobi method')

# plot exact solution
fig = plt.figure(figsize=(size, size))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, u_ex, cmap=plt.cm.jet)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('u', fontsize=16)
ax.set_title(r'$u = \sin\left(2\pi n x\right)\sin\left(2\pi n y\right)$')
# plt.show()
