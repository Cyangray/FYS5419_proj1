#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 02:50:44 2024

@author: francesco
"""

import numpy as np
import matplotlib.pyplot as plt

n_Vs = 20
Vs = np.linspace(0, 2, n_Vs)
eigvals_ana = np.zeros((n_Vs, 5))
entropy = np.zeros((n_Vs, 5))
def Hamiltonian(e, V, W):
    return np.matrix([[-2*e,          0,       np.sqrt(6)*V, 0,       0           ],
                      [0,            -e + 3*W, 0,            3*V,     0           ],
                      [np.sqrt(6)*V,  0,       4*W,          0,       np.sqrt(6)*V],
                      [0,             3*V,     0,            e + 3*W, 0         ],
                      [0,             0,       np.sqrt(6)*V, 0,       2*e         ]])


e = 1
W = 0
eigvals = np.zeros((n_Vs, 5))
eigvects = np.zeros((n_Vs, 25))
for i, V in enumerate(Vs):
    H = Hamiltonian(e, V, W)
    EigValues, EigVectors = np.linalg.eig(H)
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[permute]
    eigvals[i,:] = EigValues
    eigvects[i,:] = np.array(EigVectors.flatten())
    
def E0_HF(E,V,W, N = 2):
    v = V*(N-1)/E
    if v < 1:
        return -N/2*(E +0*V + 0*W)
    else:
        return -N/2*((E**2 + (N-1)**2*V**2)/(2*(N-1)*V) + W*0)


fig, ax = plt.subplots(figsize = (6,4))
ax.plot(Vs, eigvals[:,0], label = r'$\epsilon_0$')
ax.plot(Vs, eigvals[:,1], label = r'$\epsilon_1$')
ax.plot(Vs, eigvals[:,2], label = r'$\epsilon_2$')
ax.plot(Vs, eigvals[:,3], label = r'$\epsilon_3$')
ax.plot(Vs, eigvals[:,4], label = r'$\epsilon_4$')

HF_sol = [E0_HF(e,i,W,N=4) for i in Vs]
ax.plot(Vs, HF_sol, 'b:', label = r'HF solution, $W = 0$')
ax.set_xlabel(r'$V$')
ax.set_ylabel('energy')
#ax.legend()
'''
W = 0.5
eigvals = np.zeros((n_Vs, 5))
eigvects = np.zeros((n_Vs, 25))
for i, V in enumerate(Vs):
    H = Hamiltonian(e, V, W)
    EigValues, EigVectors = np.linalg.eig(H)
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[permute]
    eigvals[i,:] = EigValues
    eigvects[i,:] = np.array(EigVectors.flatten())


fig2, ax2 = plt.subplots()
ax2.plot(Vs, eigvals[:,0], label = r'$\epsilon_0$')
ax2.plot(Vs, eigvals[:,1], label = r'$\epsilon_1$')
ax2.plot(Vs, eigvals[:,2], label = r'$\epsilon_2$')
ax2.plot(Vs, eigvals[:,3], label = r'$\epsilon_3$')
ax2.plot(Vs, eigvals[:,4], label = r'$\epsilon_4$')

ax2.set_xlabel(r'$V$')
ax2.set_ylabel('energy')
ax2.legend()
'''


#qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator as AerEstimator
from tqdm import tqdm

E = 1
W = 0

#Use Qiskit's integrated numpy eigensolver as a check
from qiskit_algorithms import NumPyMinimumEigensolver
numpy_solver = NumPyMinimumEigensolver()
Vs_num = np.linspace(0,2,num=10)
numpy_eigs = np.zeros(len(Vs_num))
for i, V in enumerate(tqdm(Vs_num)):
    observable = SparsePauliOp.from_list([("ZI", E),
                                          ("IZ", E),
                                          ("XI", np.sqrt(6)*V/2),
                                          ("IX", np.sqrt(6)*V/2),
                                          ("ZX", np.sqrt(6)*V/2),
                                          ("XZ", -np.sqrt(6)*V/2)]
                                         )
    result = numpy_solver.compute_minimum_eigenvalue(operator=observable)
    numpy_eigs[i] = result.eigenvalue.real
    
#set up Aer estimator. The estimator runs 1024 simulations of the circuit and returns the expectation value (number of '1' divided by number of shots).
noiseless_estimator = AerEstimator(run_options={"shots": 1024})

def store_intermediate_result(eval_count, parameters, mean, std):
    #callback function for introspection
    counts.append(eval_count)
    values.append(mean)

#set up VQE
iterations = 200
params = ParameterVector('theta', 2)
ansatz = QuantumCircuit(2)
ansatz.ry(params[0], 0)
ansatz.cry(params[1], 0,1)
ansatz.draw('mpl')
optimizer = SPSA(maxiter=iterations) #optimizer
vqe = VQE(noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result) #VQE class instantiated

#evaluate eigenvalue for 10 lambdas
Vs2 = np.linspace(0,2,num=10)
VQE_res = np.zeros(len(Vs2))
counts_list = []
values_list = []
for i, V in enumerate(tqdm(Vs2)):
    counts = []
    values = []
    observable = SparsePauliOp.from_list([("ZI", E),
                                          ("IZ", E),
                                          ("XI", np.sqrt(6)*V/2),
                                          ("IX", np.sqrt(6)*V/2),
                                          ("ZX", np.sqrt(6)*V/2),
                                          ("XZ", -np.sqrt(6)*V/2)]
                                         )
    result = vqe.compute_minimum_eigenvalue(operator=observable)
    VQE_res[i] = result.eigenvalue.real
    counts_list.append(counts.copy())
    values_list.append(values.copy())
    
#plot results
ax.plot(Vs_num, numpy_eigs, 'bo', label = 'Numpy min eigensolver')
ax.plot(Vs2, VQE_res, 'rx', label = 'Qiskit VQE')
ax.set_xlabel(r'$V$')
ax.set_ylabel('energy')
ax.legend()
fig.tight_layout()
fig.savefig('J2_W0_VQE.pdf', format = 'pdf')

#plot introspection
fig2, ax2 = plt.subplots()
ax2.plot(counts, values)