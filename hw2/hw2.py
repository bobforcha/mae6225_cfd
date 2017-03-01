__author__ = 'Bob Forcha'
"""
hw2.py - a script to complete homework 2 for MAE 6225 - Computational Fluid Dynamics at The George Washington
University.

Discretization of the Poisson equation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create mesh grid
n = 128
h = 1.0/n
x_start, x_end = -h/2.0, 1.0 + h/2.0
y_start, y_end = -1.0/(2.0*n), 1.0 + 1.0/(2.0*n)
x = np.linspace(x_start, x_end, n+2)
y = np.linspace(y_start, y_end, n+2)
X, Y = np.meshgrid(x, y)

# test plot
size = 5
plt.figure(figsize=(size, size))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.grid(True)
plt.title('Grid Points', fontsize=20, fontweight='bold')
plt.scatter(X[1:n+1, 1:n+1], Y[1:n+1, 1:n+1], s=20, color='#FF0D0D', marker='o', )
plt.scatter(X[:, 0], Y[:, 0], s=20, color='k', marker='o', label='Ghost points')
plt.scatter(X[:, n+1], Y[:, n+1], s=20, color='k', marker='o')
plt.scatter(X[0, 1:n+1], Y[0, 1:n+1], s=20, color='k', marker='o')
plt.scatter(X[n+1, 1:n+1], Y[n+1, 1:n+1], s=20, color='k', marker='o')
plt.legend(loc='best')

# exact solution for part 2
u_ex = np.sin(2*math.pi*n*X[1:n+1, 1:n+1])*np.sin(2*math.pi*n*Y[1:n+1, 1:n+1])

# plot exact solution
fig = plt.figure(figsize=(size, size))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X[1:n+1, 1:n+1], Y[1:n+1, 1:n+1], u_ex, cmap=plt.cm.jet)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('u', fontsize=16)
ax.set_title(r'Exact solution to $\nabla^2 u = f\left(x, y\right)$')
plt.show()