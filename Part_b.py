#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:33:58 2024

@author: francesco
"""

import matplotlib.pyplot as plt
import numpy as np
dim = 2
Hamiltonian = np.zeros((dim,dim))
E0 = 0.0
E1 = 4.0
Vnondiag = 0.20
Vdiag = 3.0
Eigenvalue = np.zeros(dim)
# setting up the Hamiltonian in terms of the identity matrix and the Pauli matrices X and Z
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
# identity matrix
I = np.array([[1,0],[0,1]])

epsilon = (E0+E1)*0.5
omega = (E0-E1)*0.5
c = 0.0
omega_z = Vdiag
omega_x = Vnondiag
H0 = epsilon*I + omega*Z
H1 = c*I+omega_z*Z+omega_x*X
n_lbds = 50
lbds = np.linspace(0,1, num = n_lbds)
eigvals = np.zeros((n_lbds, 2))
eigvects = np.zeros((n_lbds, 4))
for i, lbd in enumerate(lbds):
    Hamiltonian = H0 + lbd*H1
    EigValues, EigVectors = np.linalg.eig(Hamiltonian)
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[permute]
    eigvals[i,:] = EigValues
    eigvects[i,:] = np.array(EigVectors.flatten())
    
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(lbds, eigvals[:,0], label = r'$\epsilon_0$')
ax.plot(lbds, eigvals[:,1], label = r'$\epsilon_1$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('energy')
ax.legend()
fig.tight_layout()
fig.savefig('part_b_eigvals.pdf', format = 'pdf')

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(lbds, eigvects[:,0]**2, label = r'$|0\rangle$')
ax2.plot(lbds, eigvects[:,1]**2, label = r'$|1\rangle$')
ax2.legend()
ax2.set_xlabel(r'$\lambda$')
fig2.tight_layout()
fig2.savefig('part_b_eigvects.pdf', format = 'pdf')