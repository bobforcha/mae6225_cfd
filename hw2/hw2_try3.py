__author__ = 'Bob Forcha'

"""
Attempt to numerically approximate an exact solution to the Poisson equation
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# create mesh grid to represent domain
ni, nj = 7, 7
n = range(1, int((np.minimum(ni, nj) - 1)/2))
dx, dy = 1/(ni-1), 1/(nj-1)
x_start, x_end = -dx/2, 1 + dx/2
y_start, y_end = -dy/2, 1 + dy/2
x = np.linspace(x_start, x_end, ni)
y = np.linspace(y_start, y_end, nj)
X, Y = np.meshgrid(x, y)

# exact solution at n=1
u_ex = np.sin(2*math.pi*n[0]*X)*np.sin(2*math.pi*n[0]*Y)
rhs = -8*math.pi**2*n[0]**2*u_ex

# plot grid and exact solution
size = 5
plt.figure(figsize=(size, size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(-dx, 1 + dx)
plt.ylim(-dx, 1 + dx)
plt.grid(True)
plt.title('Grid Points', fontsize=20, fontweight='bold')
plt.scatter(X[1:nj-1, 1:ni-1], Y[1:nj-1, 1:ni-1], s=20, color='#FF0D0D', marker='o', )
plt.scatter(X[:, 0], Y[:, 0], s=20, color='k', marker='o', label='Ghost points')
plt.scatter(X[:, ni-1], Y[:, ni-1], s=20, color='k', marker='o')
plt.scatter(X[0, 1:ni-1], Y[0, 1:ni-1], s=20, color='k', marker='o')
plt.scatter(X[nj-1, 1:ni-1], Y[nj-1, 1:ni-1], s=20, color='k', marker='o')
plt.legend(loc='best')

# plot exact solution
fig = plt.figure(figsize=(size, size))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, u_ex, cmap=plt.cm.jet)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('u', fontsize=16)
ax.set_title(r'$u = \sin\left(2\pi n x\right)\sin\left(2\pi n y\right)$')

# jacobi method
u = np.zeros(np.shape(X))
n_max = 1e2
count = 0

while count < n_max:
    u_old = u
    for j in range(1, nj-1):
        for i in range(1, ni-1):
            u[j, i] = 1/4*(u_old[j, i-1] + u_old[j, i+1] + u_old[j-1, i] + u_old[j+1, i]) - (dx*dy)/4*rhs[j, i]

    u[0, :] = -u[1, :]
    u[nj-1, :] = -u[nj-2, :]
    u[:, 0] = -u[:, 1]
    u[:, ni-1] = -u[:, ni-2]
    count += 1

# plot jacobi solution
fig = plt.figure(figsize=(size, size))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, u, cmap=plt.cm.jet)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('u', fontsize=16)
ax.set_title(r'$u = \sin\left(2\pi n x\right)\sin\left(2\pi n y\right)$')
plt.show();
