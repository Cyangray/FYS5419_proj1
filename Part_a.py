#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:47:19 2024

@author: francesco
"""

#Problem 1a

import numpy as np
import qiskit as qk
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

#First part: define 1 qubit basis, and use the X, Y, Z, H and P gates on it.
def SetupBasis(z):
    '''
    Parameters
    ----------
    z : int
        value of the first element in a 2x1 matrix representing a qubit. It can
        be 0 or 1. if 0, the second element will be 1, if 1, the other will be 0

    Returns
    -------
    qubit: np.array

    '''
    if z == 0:
        basis = np.array([1,0])
        basis.shape = (2,1)
        return basis
    elif z == 1:
        basis = np.array([0,1])
        basis.shape = (2,1)
        return basis
    

#define gates
I = np.eye(2) #identity matrix
H = np.array([[1,1],[1,-1]])/np.sqrt(2)
P = np.array([[1,0],[0,1j]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
PauliMatrices = [X, Y, Z]
PauliMatricesLabels = ['X', 'Y', 'Z']


for initial_state in [0,1]:
    #apply the Pauli matrices gates
    if initial_state == 0:
        print('For initial state in basis |0> = [1,0]:')
    else:
        print('For initial state in basis |1> = [0,1]:')
    
    for PauliMatrix, label in zip(PauliMatrices, PauliMatricesLabels):
        result = PauliMatrix @ SetupBasis(initial_state)
        print('Apply the ' + label + ' gate:')
        print(result)
    
    #apply the Hadamard gate
    print('Apply the H gate:')
    print(H @ SetupBasis(initial_state))
    
    #apply the phase gate
    print('Apply the P gate:')
    print(P @ SetupBasis(initial_state))
    print('\n')

    

    
#Second part: define Bell states, and deploy Hadamard + CNOT gates on Bell state
#measure several times

#define two basis states:
zero = SetupBasis(0)
one = SetupBasis(1)

#there are four bell states:
phi_plus = (np.kron(zero, zero) + np.kron(one,one))/np.sqrt(2)
phi_minus = (np.kron(zero, zero) - np.kron(one,one))/np.sqrt(2)
psi_plus = (np.kron(zero, one) + np.kron(one,zero))/np.sqrt(2)
psi_minus = (np.kron(zero, one) - np.kron(one,zero))/np.sqrt(2)

#otherwise, phi_plus can be also defined as a |00> state, where a Hadamard and
#a CNOT states are applied
basis_state = np.kron(zero,zero)
H2 = np.kron(H,I) #2-qbits, Hadamard on first qubit
print(H2)
CNOT = np.zeros((4,4))
CNOT[0,0] = CNOT[1,1] = CNOT[2,3] = CNOT[3,2] = 1

phi_plus2 = CNOT @ H2 @ basis_state

#How to measure states?

#projection operators for one qubit system
P0 = np.array([[1,0],[0,0]])
P1 = np.array([[0,0],[0,1]])

def density(state):
    #Calculate density matrix
    return np.outer(state, np.conj(state))

def matrix_kron_power(M, exponent):
    #calculate Kroeneker product of matrices with themselves
    result = M
    if exponent == 0:
        return 1.0
    for _ in range(int(exponent - 1)):
        result = np.kron(result, M)
    return result

def measure(psi, qubit_idx, tostate = 0, collapse = True):
    #returns the probability of reading 0 (or 1) when measuring qubit in  a 
    #quantum circuit at index qubit_idx. Function inspired by Hundts' "Quantum computing for programmers", Chapter 2.
    if tostate == 0:
        op = P0
    else:
        op = P1
    if qubit_idx > 0:
        op = np.kron(matrix_kron_power(I,qubit_idx), op)
    if qubit_idx < (len(psi)/2 - 1):
        op = np.kron(op, matrix_kron_power(I, len(psi)/2 - qubit_idx - 1))
    prob0 = np.trace(op @ density(psi)) 
    if collapse:
        mvmul = np.dot(op, psi)
        divisor = np.real(np.linalg.norm(mvmul))
        normed = mvmul / divisor
        return np.real(prob0), normed
    return np.real(prob0), psi

def yield_result(psi, qubit_idx):
    #measures, yields either a 0 or a 1, and the collapsed state given that measurement
    prob, _ = measure(psi, qubit_idx)
    result_of_measurement = np.random.binomial(1, 1-prob)
    _, collapsed_state = measure(psi, qubit_idx, tostate = result_of_measurement)
    return result_of_measurement, collapsed_state

def run_2q_simulation(psi, shots = 1000):
    #runs simulations, stores results in a numpy array, yields resulting array
    #result labels for res_array: [00, 01, 10, 11]
    res_array = np.zeros(4)
    for i in range(shots):
        q1, collapsed_state = yield_result(psi, 0)
        q2, _ = yield_result(collapsed_state, 1)
        res_array[int(q1*2 + q2)] += 1
    return res_array


Bell_simulation = run_2q_simulation(phi_plus, shots = 1000)
print(Bell_simulation)

#check with Qiskit:
qc = QuantumCircuit(2, 2) # Start a circuit with 2 qubits and 2 registers where to store results

#create the phi_plus Bell state by applying an H gate and then a CNOT gate
qc.h(0)
qc.cx(0,1)
qc.h(0)
qc.cx(0,1) 

# measure the qubits, specify which qubit you want to measure
qc.measure(0,0)
qc.measure(1,1)

qc.draw('mpl')

#simulate circuit 1000 times
simulator = AerSimulator()
result_ideal = simulator.run(qc, shots = 1000).result()
counts_ideal = result_ideal.get_counts(0)
print('Counts(ideal):', counts_ideal)

#apply Hadamard and CNOT gate on Bell state:
result = CNOT @ H2 @ phi_plus
print(result)

Bell_CNOT_H = run_2q_simulation(result, shots = 1000)
print(Bell_CNOT_H)






