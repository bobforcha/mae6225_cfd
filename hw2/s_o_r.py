__author__ = 'Bob Forcha'

"""
this script numericallyapproximates an exact solution to the Poisson equation using the Gauss-Seidel method with
Successive Over Relaxation
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
    u_ex[num, :, :] = np.sin(2*math.pi*n[num]*X)*np.sin(2*math.pi*n[num]*Y)
    rhs[num, :, :] = -8*math.pi**2*n[num]**2*u_ex[num, :, :]


# define gauss-seidel-SOR function
def gauss_seidel_sor(phi_0, f, omega,  dx, dy, err_min, n_max, phi_ex):
    err = 1
    phi = phi_0
    n = 0

    while err > err_min and n < n_max:
        phi_old = phi
        for j in range(1, nj-1):
            for i in range(1, ni-1):
                phi[j, i] = omega/4*(phi[j, i-1] + phi_old[j, i+1] + phi[j-1, i] + phi_old[j+1, i] - dx*dy*f[j, i])\
                          + (1 - omega)*phi_old[j, i]

        phi[0, :] = -phi[1, :]
        phi[nj-1, :] = -phi[nj-2, :]
        phi[:, 0] = -phi[:, 1]
        phi[:, ni-1] = -phi[:, ni-2]

        err = np.max(np.abs(phi - phi_ex)/phi_ex)
        n += 1

    return phi

# evaluate
omega_opt = 2/(2 - (np.cos(2*math.pi*dx)))
u0 = np.zeros(np.shape(u_ex))
u = u0
size = 5

for index in range(len(n)):
    u[index, :, :] = gauss_seidel_sor(u[index, :, :], rhs[index, :, :], omega_opt, dx, dy, 1e-4, 1e4, u_ex[index, :, :])

    # plot approximate solution
    fig = plt.figure(figsize=(size, size))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, u[index, :, :], color='b', alpha=0.8, label='SOR')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('u', fontsize=16)
    ax.set_title('SOR Method, n = %g' % n[index], fontsize=20)

plt.show();