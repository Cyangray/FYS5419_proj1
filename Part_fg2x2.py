#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 02:44:33 2024

@author: francesco
"""

import numpy as np
import matplotlib.pyplot as plt

E = 1.0
W = 0
V = np.linspace(0,2,20)

def eig0(E,V,W): 
    return np.sqrt(E**2 + V**2)

def eig1(E,V,W):
    return W + 0*V + 0*E

def eig2(E,V,W): 
    return - np.sqrt(E**2 + V**2)

def E0_HF(E,V,W, N = 2):
    v = V*(N-1)/E
    if v < 1:
        return -N/2*(E +0*V + 0*W)
    else:
        return -N/2*((E**2 + (N-1)**2*V**2)/(2*(N-1)*V) + W*0)

fig, ax = plt.subplots()
ax.plot(V, eig0(E,V,W), 'r-', label = r'$\lambda_0, W = 0$')
ax.plot(V, eig1(E,V,W), 'g-', label = r'$\lambda_1, W = 0$')
ax.plot(V, eig2(E,V,W), 'b-', label = r'$\lambda_2, W = 0$')
HF_sol = [E0_HF(E,i,W) for i in V]
ax.plot(V, HF_sol, 'b:', label = r'HF solution, $W = 0$')
'''
W = -0.5
ax.plot(V, eig0(E,V,W), 'r--', label = r'$\lambda_0, W = 1$')
ax.plot(V, eig1(E,V,W), 'g--', label = r'$\lambda_1, W = 1$')
ax.plot(V, eig2(E,V,W), 'b--', label = r'$\lambda_2, W = 1$')
ax.set_xlabel('V')
ax.set_ylabel('Energy')
#ax.legend()

V = 0.5
W = np.linspace(0,2,20)
fig2, ax2 = plt.subplots()
ax2.plot(W, eig0(E,V,W), 'r-', label = r'$\lambda_0, V = 0.5$')
ax2.plot(W, eig1(E,V,W), 'g-', label = r'$\lambda_1, V = 0.5$')
ax2.plot(W, eig2(E,V,W), 'b-', label = r'$\lambda_2, V = 0.5$')
V = 1.5
ax2.plot(W, eig0(E,V,W), 'r--', label = r'$\lambda_0, V = 1.5$')
ax2.plot(W, eig1(E,V,W), 'g--', label = r'$\lambda_1, V = 1.5$')
ax2.plot(W, eig2(E,V,W), 'b--', label = r'$\lambda_2, V = 1.5$')
ax2.legend()
ax2.set_xlabel('W')
ax2.set_ylabel('Energy')
'''






#qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator as AerEstimator
from tqdm import tqdm

E = 1
W = 0

#Use Qiskit's integrated numpy eigensolver as a check
from qiskit_algorithms import NumPyMinimumEigensolver
numpy_solver = NumPyMinimumEigensolver()
Vs_num = np.linspace(0,2,num=20)
numpy_eigs = np.zeros(len(Vs_num))
for i, V in enumerate(tqdm(Vs_num)):
    observable = SparsePauliOp.from_list([("Z", E),
                                          ("X", V)]
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
ansatz = TwoLocal(1, rotation_blocks=["ry"], reps=0) #two qubit, two consecutive rotation blocks
ansatz.decompose().draw('mpl')
spsa = SPSA(maxiter=iterations) #optimizer
vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result) #VQE class instantiated

#evaluate eigenvalue for 10 lambdas
Vs2 = np.linspace(0,2,num=20)
VQE_res = np.zeros(len(Vs2))
counts_list = []
values_list = []
for i, V in enumerate(tqdm(Vs2)):
    counts = []
    values = []
    observable = SparsePauliOp.from_list([("Z", E),
                                          ("X", V)]
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
fig.savefig('J1_W0_gray.pdf', format = 'pdf')

#plot introspection
fig2, ax2 = plt.subplots()
ax2.plot(counts, values)