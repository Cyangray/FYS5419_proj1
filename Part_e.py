#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:54:04 2024

@author: francesco
"""

#Numerical solution from part b
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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





#qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator as AerEstimator

#Use Qiskit's integrated numpy eigensolver as a check
from qiskit_algorithms import NumPyMinimumEigensolver
numpy_solver = NumPyMinimumEigensolver()
lbds_num = np.linspace(0,1,num=10)
numpy_eigs = np.zeros(len(lbds_num))
for i, lbd in enumerate(tqdm(lbds_num)):
    observable = SparsePauliOp.from_list([("II", (E0 + E1 + E2 + E3)/4),
                                          ("ZZ", (E0 - E1 - E2 + E3)/4 + lbd*Hz),
                                          ("ZI", (E0 + E1 - E2 - E3)/4),
                                          ("IZ", (E0 - E1 + E2 - E3)/4),
                                          ("XX", lbd*Hx)]
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
iterations = 400
ansatz = TwoLocal(2, rotation_blocks=["rx", "ry"], entanglement_blocks='cx', reps=1) #two qubit, two consecutive rotation blocks. Only rotation and one CNOT gate will converge some of the points to the first excited level.
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
    observable = SparsePauliOp.from_list([("II", (E0 + E1 + E2 + E3)/4),
                                          ("ZZ", (E0 - E1 - E2 + E3)/4 + lbd*Hz),
                                          ("ZI", (E0 + E1 - E2 - E3)/4),
                                          ("IZ", (E0 - E1 + E2 - E3)/4),
                                          ("XX", lbd*Hx)]
                                         )
    result = vqe.compute_minimum_eigenvalue(operator=observable)
    VQE_res[i] = result.eigenvalue.real
    counts_list.append(counts.copy())
    values_list.append(values.copy())
    
#plot results
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(lbds, eigvals[:,0], label = r'$\epsilon_0$')
ax.plot(lbds, eigvals[:,1], label = r'$\epsilon_1$')
ax.plot(lbds, eigvals[:,2], label = r'$\epsilon_2$')
ax.plot(lbds, eigvals[:,3], label = r'$\epsilon_3$')
ax.plot(lbds_num, numpy_eigs, 'bo', label = 'Numpy min eigensolver')
ax.plot(lbds2, VQE_res, 'rx', label = 'Qiskit VQE')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('energy')
ax.legend()
fig.tight_layout()
fig.savefig('part_e_eigvals.pdf', format = 'pdf')

#plot introspection
fig2, ax2 = plt.subplots()
ax2.plot(counts, values)
