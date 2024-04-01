#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:33:24 2024

@author: francesco
"""

import numpy as np
import matplotlib.pyplot as plt


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
E = 1.0
W = 0
V = np.linspace(0,2,20)
fig, ax = plt.subplots()
ax.plot(V, eig0(E,V,W), 'r-', label = r'$\lambda_0, W = 0$')
ax.plot(V, eig1(E,V,W), 'g-', label = r'$\lambda_1, W = 0$')
ax.plot(V, eig2(E,V,W), 'b-', label = r'$\lambda_2, W = 0$')
HF_sol = [E0_HF(E,i,W) for i in V]
ax.plot(V, HF_sol, 'b:', label = r'HF solution, $W = 0$')
W = -1.5
ax.plot(V, eig0(E,V,W), 'r--', label = r'$\lambda_0, W = -1.5$')
ax.plot(V, eig1(E,V,W), 'g--', label = r'$\lambda_1, W = -1.5$')
ax.plot(V, eig2(E,V,W), 'b--', label = r'$\lambda_2, W = -1.5$')
ax.set_xlabel('V')
ax.set_ylabel('Energy')
ax.legend()
fig.tight_layout()
fig.savefig('J1_V_W.pdf', format = 'pdf')


#qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator as AerEstimator
from tqdm import tqdm

#Use Qiskit's integrated numpy eigensolver as a check
from qiskit_algorithms import NumPyMinimumEigensolver
numpy_solver = NumPyMinimumEigensolver()
Vs_num = np.linspace(0,2,num=20)
numpy_eigs = np.zeros(len(Vs_num))
for i, V in enumerate(tqdm(Vs_num)):
    observable = SparsePauliOp.from_list([("IZ", E/2),
                                          ("ZI", E/2),
                                          ("XX", (W-V)/2),
                                          ("YY", (W+V)/2)]
                                         )
    result = numpy_solver.compute_minimum_eigenvalue(operator=observable)
    numpy_eigs[i] = result.eigenvalue.real

#set up Aer estimator. The estimator runs 1024 simulations of the circuit and returns the expectation value (number of '1' divided by number of shots).
noiseless_estimator = AerEstimator(run_options={"shots": 1.5*1024})

def store_intermediate_result(eval_count, parameters, mean, std):
    #callback function for introspection
    counts.append(eval_count)
    values.append(mean)

#set up VQE
iterations = 400
ansatz = TwoLocal(2, rotation_blocks=["rx", "ry"], entanglement_blocks='cx', reps=1) #two qubit, two consecutive rotation blocks
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
    observable = SparsePauliOp.from_list([("IZ", E/2),
                                          ("ZI", E/2),
                                          ("XX", (W+V)/2),
                                          ("YY", (W-V)/2)]
                                         )
    result = vqe.compute_minimum_eigenvalue(operator=observable)
    VQE_res[i] = result.eigenvalue.real
    counts_list.append(counts.copy())
    values_list.append(values.copy())

#plot results
V = np.linspace(0,2,20)
W = -1.5
fig2, ax2 = plt.subplots(figsize = (6,4))
ax2.plot(V, eig0(E,V,W), 'r--', label = r'$\lambda_0, W = -1.5$')
ax2.plot(V, eig1(E,V,W), 'g--', label = r'$\lambda_1, W = -1.5$')
ax2.plot(V, eig2(E,V,W), 'b--', label = r'$\lambda_2, W = -1.5$')
ax2.plot(Vs_num, numpy_eigs, 'bo', label = 'Numpy min eigensolver')
ax2.plot(Vs2, VQE_res, 'rx', label = 'Qiskit VQE')
ax2.set_xlabel(r'$V$')
ax2.set_ylabel('energy')
ax2.legend()
fig2.tight_layout()
fig2.savefig('J1_W15_vqe.pdf', format = 'pdf')

#plot introspection
fig2, ax2 = plt.subplots()
ax2.plot(counts, values)
