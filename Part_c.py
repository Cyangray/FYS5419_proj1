#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:15:39 2024

@author: francesco
"""

#Numerical solution from part b
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

#qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator as AerEstimator

#Use Qiskit's integrated numpy eigensolver as a check
from qiskit_algorithms import NumPyMinimumEigensolver
numpy_solver = NumPyMinimumEigensolver()
lbds_num = np.linspace(0,1,num=10)
numpy_eigs = np.zeros(len(lbds_num))
for i, lbd in enumerate(tqdm(lbds_num)):
    observable = SparsePauliOp.from_list([("I", epsilon + lbd*c),("X", lbd*omega_x), ("Z", omega + lbd*omega_z)])
    result = numpy_solver.compute_minimum_eigenvalue(operator=observable)
    numpy_eigs[i] = result.eigenvalue.real
    
#set up Aer estimator. The estimator runs 1024 simulations of the circuit and returns the expectation value (number of '1' divided by number of shots).
noiseless_estimator = AerEstimator(run_options={"shots": 1024})

def store_intermediate_result(eval_count, parameters, mean, std):
    #callback function for introspection
    counts.append(eval_count)
    values.append(mean)

#set up VQE
iterations = 125
ansatz = TwoLocal(1, rotation_blocks=["rx", "ry"], reps=0) #one qubit, two consecutive rotation blocks
ansatz.decompose().draw('mpl')
spsa = SPSA(maxiter=iterations) #optimizer
vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result) #VQE class instantiated

#evaluate eigenvalue for 10 lambdas
lbds2 = np.linspace(0,1,num=10)
VQE_res = np.zeros(len(lbds2))
counts_list = []
values_list = []
for i, lbd in enumerate(tqdm(lbds2)):
    counts = []
    values = []
    observable = SparsePauliOp.from_list([("I", epsilon + lbd*c),("X", lbd*omega_x), ("Z", omega + lbd*omega_z)])
    result = vqe.compute_minimum_eigenvalue(operator=observable)
    VQE_res[i] = result.eigenvalue.real
    counts_list.append(counts.copy())
    values_list.append(values.copy())
    
#plot results
fig, ax = plt.subplots(figsize = (6,4))
ax.plot(lbds, eigvals[:,0], label = r'$\epsilon_0$')
ax.plot(lbds, eigvals[:,1], label = r'$\epsilon_1$')
ax.plot(lbds_num, numpy_eigs, 'bo', label = 'Numpy min eigensolver')
ax.plot(lbds2, VQE_res, 'rx', label = 'Qiskit VQE')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('energy')
ax.legend()
fig.tight_layout()
fig.savefig('part_c_eigvals.pdf', format = 'pdf')

#plot introspection
fig2, ax2 = plt.subplots()
ax2.plot(counts, values)





