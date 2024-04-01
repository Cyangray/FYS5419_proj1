#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:40:10 2024

@author: francesco
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm, expm

# setting up the Hamiltonian in terms of the identity matrix and the Pauli matrices X and Z
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
# identity matrix
I = np.array([[1,0],[0,1]])

dim = 4
H0 = np.zeros((dim,dim))
E0 = 0.0
E1 = 2.5
E2 = 6.5
E3 = 7.0
Hx = 2.0
Hz = 3.0

H0[0,0] = E0
H0[1,1] = E1
H0[2,2] = E2
H0[3,3] = E3

HI = Hx * np.kron(X,X) + Hz * np.kron(Z,Z)

#projection operators for one qubit system
P0 = np.array([[1,0],[0,0]])
P1 = np.array([[0,0],[0,1]])
Proj_A = np.kron(P0, I)
Proj_B = np.kron(P1, I)

def density(state):
    #Calculate density matrix
    return np.outer(state, np.conj(state))

def log2M(a): # base 2 matrix logarithm
    return logm(a)/np.log(2.0)

def TraceOutSingle(rho, index):
    """Trace out single qubit from density matrix. Adapted from Hundt's "quantum computing for programmers" """
    nbits = int(np.log2(rho.shape[0]))
    if index > nbits:
        raise AssertionError(
            'Error in TraceOutSingle, invalid index (>nbits).')
    zero = np.array([1.0, 0.0])
    one = np.array([0.0, 1.0])
    
    p0 = p1 = np.array([[1.0]])
    for idx in range(nbits):
        if idx == index:
            p0 = np.kron(p0, zero)
            p1 = np.kron(p1, one)
        else:
            p0 = np.kron(p0, I)
            p1 = np.kron(p1, I)
    rho0 = p0 @ rho
    rho0 = rho0 @ p0.transpose()
    rho1 = p1 @ rho
    rho1 = rho1 @ p1.transpose()
    rho_reduced = rho0 + rho1
    return rho_reduced

def calc_entropy(eigenvectors, state):
    rho = density(EigVectors[:,state])
    red_rho = TraceOutSingle(rho, state)
    return -np.trace(red_rho @ log2M(red_rho))
    

n_lbds = 50
lbds = np.linspace(0,1, num = n_lbds)
eigvals = np.zeros((n_lbds, 4))
eigvects = np.zeros((n_lbds, 16))
entropies = np.zeros(n_lbds)
entropies2 = np.zeros(n_lbds)
entropiestot = np.zeros(n_lbds)
entropy_state = 1
for i, lbd in enumerate(lbds):
    Hamiltonian = H0 + lbd*HI
    EigValues, EigVectors = np.linalg.eig(Hamiltonian)
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[:,permute]
    eigvals[i,:] = EigValues
    eigvects[i,:] = np.array(EigVectors.flatten())
    entropies[i] = calc_entropy(EigVectors, 0)
    
    
    
    
fig, ax = plt.subplots()
ax.plot(lbds, eigvals[:,0], label = r'$\epsilon_0$')
ax.plot(lbds, eigvals[:,1], label = r'$\epsilon_1$')
ax.plot(lbds, eigvals[:,2], label = r'$\epsilon_2$')
ax.plot(lbds, eigvals[:,3], label = r'$\epsilon_3$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('energy')
ax.legend()

fig2, ax2 = plt.subplots()
ax2.plot(lbds, entropies)
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel('g.s. entropy')
fig2.tight_layout()
fig2.savefig('entropies.pdf', format = 'pdf')

'''
fig2, ax2 = plt.subplots()
ax2.plot(lbds, eigvects[:,0]**2, label = r'$|00\rangle$')
ax2.plot(lbds, eigvects[:,1]**2, label = r'$|01\rangle$')
ax2.plot(lbds, eigvects[:,2]**2, label = r'$|10\rangle$')
ax2.plot(lbds, eigvects[:,3]**2, label = r'$|11\rangle$')
ax2.legend()
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r'$\langle \psi_{eig} |$ overlap')
'''