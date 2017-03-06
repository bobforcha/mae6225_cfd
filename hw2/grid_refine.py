"""
this script performs a grid refinement study of the Jacobi and SOR methods
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# define jacobi function
def jacobi(phi_0, f, ni, nj, dx, dy, err_min, n_max, phi_ex):
    err_mag = 1
    phi = phi_0
    n = 0

    while err_mag > err_min and n < n_max:
        phi_old = phi
        for j in range(1, nj-1):
            for i in range(1, ni-1):
                phi[j, i] = 1/4*(phi_old[j, i-1] + phi_old[j, i+1] + phi_old[j-1, i] + phi_old[j+1, i])\
                            - (dx*dy)/4*f[j, i]

        phi[0, :] = -phi[1, :]
        phi[-1, :] = -phi[-2, :]
        phi[:, 0] = -phi[:, 1]
        phi[:, -1] = -phi[:, -2]

        err = phi - phi_ex
        err_mag = np.sum(err**2)
        n += 1

    return phi, err_mag


# define gauss-seidel-SOR function
def gauss_seidel_sor(phi_0, f, omega, ni, nj, dx, dy, err_min, n_max, phi_ex):
    err_mag = 1
    phi = phi_0
    n = 0

    while err_mag > err_min and n < n_max:
        phi_old = phi
        for j in range(1, nj-1):
            for i in range(1, ni-1):
                phi[j, i] = omega/4*(phi[j, i-1] + phi_old[j, i+1] + phi[j-1, i] + phi_old[j+1, i] - dx*dy*f[j, i])\
                          + (1 - omega)*phi_old[j, i]

        phi[0, :] = -phi[1, :]
        phi[-1, :] = -phi[-2, :]
        phi[:, 0] = -phi[:, 1]
        phi[:, -1] = -phi[:, -2]

        err = phi - phi_ex
        err_mag = np.sum(err**2)
        n += 1

    return phi, err_mag

# main loop
k_list = np.array([5, 9, 17, 33, 65, 129, 257, 513])
err_jac = np.zeros(np.shape(k_list))
err_sor = np.zeros(np.shape(k_list))
count = 0

for k in k_list:
    ni, nj = k, k
    dx, dy = 1 / (ni - 1), 1 / (nj - 1)
    x_start, x_end = -dx / 2, 1 + dx / 2
    y_start, y_end = -dy / 2, 1 + dy / 2
    x = np.linspace(x_start, x_end, ni)
    y = np.linspace(y_start, y_end, nj)
    X, Y = np.meshgrid(x, y)
    n = 1

    u_ex = np.sin(2*math.pi*n*X)*np.sin(2*math.pi*n*Y)
    rhs = -8*math.pi**2*n**2*u_ex
    u0 = np.zeros(np.shape(X))

    u_jac, err_jac[count] = jacobi(u0, rhs, ni, nj, dx, dy, 1e-4, 10, u_ex)
    u_sor, err_sor[count] = gauss_seidel_sor(u0, rhs, 2/(2 - np.cos(2*math.pi*dx)), ni, nj, dx, dy, 1e-4, 5, u_ex)
    count += 1

# plot errors
size = 5
plt.figure(figsize=(size, size))
plt.loglog(k_list, err_jac, label='Jacobi Method', color='r', linewidth=2, linestyle='-')
plt.loglog(k_list, err_sor, label='SOR Method', color='b', linewidth=2, linestyle='-')
plt.legend(loc='best')
plt.xlabel('n', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.title('Error vs. # of Grid Points', fontsize=20)
plt.xlim(1e1, 1e3)
plt.ylim(1e-4, 1e1)
plt.grid(True)
print(err_jac)
print(err_sor)
plt.show()

