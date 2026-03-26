
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter,ParameterExpression
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.circuit import Parameter
import numpy as np
from math import atan2


parameters = []

def XGate(S, qubit):
    for i in range(len(S)):
        stateAsList = list(S[i])
        if stateAsList[qubit] == '1':
            stateAsList[qubit] = '0'
        else:
            stateAsList[qubit] = '1'
        S[i] = ''.join(stateAsList)
# Ændrer S
def CXGate(S, control, target):
    for i in range(len(S)):
        stateAsList = list(S[i])
        if stateAsList[control] == '1' and stateAsList[target] == '0':
            stateAsList[target] = '1'
        elif stateAsList[control] == '1' and stateAsList[target] == '1':
            stateAsList[target] = '0'
        S[i] = ''.join(stateAsList)
# ----------------------------------
def Algorithm1(S, coeffs):
    noq = len(S[0])
    qc = QuantumCircuit(noq)
    Scopy = S.copy()
    # Lav dif_qubits og dif_values
    T = Scopy.copy()
    dif_qubits = []
    dif_values = []
    while len(T) > 1:
        maxDiff, qubitForMaxDiff = -1, 0
        for qubit in range(0, noq):
            zeroCount, oneCount = 0, 0
            for state in T:
                if (state[qubit] == '0'):
                    zeroCount += 1
                else:
                    oneCount += 1
            if (abs(zeroCount - oneCount) > maxDiff and zeroCount != 0 and oneCount != 0):
                maxDiff = abs(zeroCount - oneCount)
                qubitForMaxDiff = qubit
        dif_qubits.append(qubitForMaxDiff)
        T0, T1 = [], []
        for state in T:
            if (state[qubitForMaxDiff] == '0'):
                T0.append(state)
            else:
                T1.append(state)
        if len(T0) < len(T1):
            T = T0.copy()
            dif_values.append(0)
        else:
            T = T1.copy()
            dif_values.append(1)
    dif = dif_qubits.pop()
    dif_values.pop()
    x1 = T[0]
    # Lav T'
    Tprime = []
    for state in S:
        allQubitValuesMatch = True
        for i in range(0, len(dif_qubits)):
            if int(state[dif_qubits[i]]) != dif_values[i]:
                allQubitValuesMatch = False
                break
        if allQubitValuesMatch == True and state != x1:
            Tprime.append(state)
    while len(Tprime) > 1:
        maxDiff, qubitNumberForMaxDiff = -1, 0
        for qubit in range(0, noq):
            zeroCount = 0
            oneCount = 0
            for state in Tprime:
                if (state[qubit] == '0'):
                    zeroCount += 1
                else:
                    oneCount += 1
            if (abs(zeroCount - oneCount) > maxDiff and zeroCount != 0 and oneCount != 0):
                maxDiff = abs(zeroCount - oneCount)
                qubitNumberForMaxDiff = qubit
        dif_qubits.append(qubitNumberForMaxDiff)
        T0, T1 = [], []
        for state in Tprime:
            if (state[qubitNumberForMaxDiff] == '0'):
                T0.append(state)
            else:
                T1.append(state)
        if len(T0) < len(T1):
            Tprime = T0.copy()
            dif_values.append(0)
        else:
            Tprime = T1.copy()
            dif_values.append(1)
    # Ændring af tilstande begynder
    x2 = Tprime[0]
    if int(x1[dif]) == 0:
        qc.x(dif)
        XGate(Scopy, dif)
    for i in range(0, noq):
        if i != dif and x1[i] != x2[i]:
            qc.cx(dif, i)
            CXGate(Scopy, dif, i)
    for qubit in dif_qubits:
        if int(x2[qubit]) == 0:
            qc.x(qubit)
            XGate(Scopy, qubit)
    # x1 er den med |1>. x2 er den med |0>
    # idx1 er indekset for x1 i S. Tilsvarende for idx2
    idx1, idx2 = 0, 0
    for i in range(len(S)):
        if S[i] == x1:
            idx1 = i
        if S[i] == x2:
            idx2 = i
    cx1, cx2 = coeffs[idx1], coeffs[idx2]
    rotationAngle = -2 * atan2(cx1, cx2)
    if len(dif_qubits) != 0:
        #############
        # RY gate addition 
        #############
        g = Parameter(f"r{len(parameters)}")
        ryGate = RYGate(g).control(len(dif_qubits))
        qc.append(ryGate, dif_qubits + [dif])
        parameters.append(g)
    elif len(dif_qubits) == 0:
        g = Parameter(f"r{len(parameters)}")
        qc.ry(g, dif)
        parameters.append(g)
    Sprime = [Scopy[idx2]]
    newCoeffs = [np.sqrt(cx1 * cx1 + cx2 * cx2)]
    for i in range(0, len(Scopy)):
        if i != idx1 and i != idx2:
            Sprime.append(Scopy[i])
            newCoeffs.append(coeffs[i])
    return qc, Sprime, newCoeffs , parameters , rotationAngle
# inputStates skal være en liste med basistilstandene for den tilstand, der ønskes forberedt
# inputCoeffs[i] skal være koefficienten, der knytter sig til basistilstanden inputStates[i]
# IMPLEMNTERINGEN KAN KUN HÅNDTERE REELLE TAL SOM KOEFFICIENTER
def Algorithm2(inputStates, inputCoeffs):
    noq = len(inputStates[0])
    # Tjek at længden af inputStates og inputCoeffs er den samme
    if len(inputStates) != len(inputCoeffs):
        #print("Fejl. Antallet af koefficienter er ikke det samme som antallet af tilstande.")
        print("Error. The number of coefficients is not the same as the number of states.")
        exit()
    # Tjek at alle tilstandene i inputStates er forskellige
    setOfStates = []
    for state in inputStates:
        stateAsInt = int(state, 2)
        if stateAsInt in setOfStates:
            #print("Fejl. Tilstanden", state, "gentager sig.")
            print("Error. The state", state, "is repeated")
            exit()
        setOfStates.append(stateAsInt)
    # Tjek at sandsynlighederne summer til 1
    pSum = 0
    for coeff in inputCoeffs:
        pSum += coeff * coeff
    if (abs(pSum - 1.0) > 1.0e-3):
        #print("Fejl. Summen af kvadratet af koefficienterne giver ikke 1.")
        print("Fejl. Sum of square of the coefficent is not equale to 1 ")
        exit()
    # Lav en kopi af inputStates for ikke at ændre inputStates selv,
    # og flip rækkefølgen af bits for hver basistilstand for
    # at det passer med qiskits big endian-repræsentation af bitstrenge
    # eller sådan noget ...
    S = []
    for i in range(len(inputStates)):
        S.append(inputStates[i][::-1])
    # Lav en kopi af inputCoeffs
    coeffs = inputCoeffs.copy()
    qc = QuantumCircuit(noq)
    valueParam = []
    while len(S) > 1:
        Chat, S, coeffs,prms, rot = Algorithm1(S, coeffs)
        qc.compose(Chat, inplace = True)
        valueParam.append(rot)
    for i in range(0, noq):
        if S[0][i] == '1':
            qc.x(i)
    return qc.inverse(), prms , valueParam







def single_excitation_efficient(
    k: int, i: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for single excitation.

    Implementation of the following operator,

    .. math::
       $\boldsymbol{U} = \exp\left(\theta\hat{a}^\dagger_k\hat{a}_i\right)$

    #. 10.1103/PhysRevA.102.062612, Fig. 3 and Fig. 8
    #. 10.1038/s42005-021-00730-0, Fig. 1

    Args:
        k: Weakly occupied spin orbital index.
        i: Strongly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Single excitation circuit.
    """
    #k = f2q(k, num_orbs)
    #i = f2q(i, num_orbs)
    if k <= i:
        raise ValueError(f"k={k}, must be larger than i={i}")
    if k - 1 == i:
        qc.rz(np.pi / 2, i)
        qc.rx(np.pi / 2, i)
        qc.rx(np.pi / 2, k)
        qc.cx(i, k)
        qc.rx(theta, i)
        qc.rz(theta, k)
        qc.cx(i, k)
        qc.rx(-np.pi / 2, k)
        qc.rx(-np.pi / 2, i)
        qc.rz(-np.pi / 2, i)
    else:
        qc.cx(k, i)
        for t in range(k - 2, i, -1):
            qc.cx(t + 1, t)
        qc.cz(i + 1, k)
        qc.ry(theta, k)
        qc.cx(i, k)
        qc.ry(-theta, k)
        qc.cx(i, k)
        qc.cz(i + 1, k)
        for t in range(i + 1, k - 1):
            qc.cx(t + 1, t)
        qc.cx(k, i)
    return qc


def double_excitation_efficient(
    k: int, l: int, i: int, j: int, num_orbs: int, qc: QuantumCircuit, theta: Parameter | ParameterExpression
) -> QuantumCircuit:
    r"""Exact circuit for double excitation.

    Implementation of the following operator,

    .. math::
       \boldsymbol{U} = \exp\left(\theta\hat{a}^\dagger_k\hat{a}^\dagger_l\hat{a}_j\hat{a}_i\right)

    #. 10.1103/PhysRevA.102.062612, Fig. 6, Fig. 7, and, Fig. 9
    #. 10.1038/s42005-021-00730-0, Fig. 2

    Args:
        k: Weakly occupied spin orbital index.
        l: Weakly occupied spin orbital index.
        i: Strongly occupied spin orbital index.
        j: Strongly occupied spin orbital index.
        num_orbs: Number of spatial orbitals.
        qc: Quantum circuit.
        theta: Circuit parameter.

    Returns:
        Double excitation circuit.
    """
    if k < i or k < j:
        raise ValueError(f"Operator only implemented for k, {k}, larger than i, {i}, and j, {j}")
    if l < i or l < j:
        raise ValueError(f"Operator only implemented for l, {l}, larger than i, {i}, and j, {j}")
    n_alpha = 0
    n_beta = 0
    if i % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if j % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if k % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if l % 2 == 0:
        n_alpha += 1
    else:
        n_beta += 1
    if n_alpha % 2 != 0 or n_beta % 2 != 0:
        raise ValueError("Operator only implemented for spin conserving operators.")
    fac = 1
    if k % 2 == l % 2 and k % 2 == 0 and i % 2 != 0:
        fac *= -1
    #k = f2q(k, num_orbs)
    #l = f2q(l, num_orbs)
    #i = f2q(i, num_orbs)
    #j = f2q(j, num_orbs)
    if k > l:
        l, k = k, l
        fac *= -1
    if i > j:
        j, i = i, j
        fac *= -1
    if l < j:
        l, j = j, l
        fac *= -1
    if k < i:
        k, i = i, k
        fac *= -1
    # cnot ladder is easier to implement if the indices are sorted.
    i_z, k_z, j_z, l_z = np.sort((k, l, i, j))
    theta = 2 * theta * fac

    qc.cx(l, k)
    qc.cx(j, i)
    qc.cx(l, j)

    if l_z != j_z + 1:
        for t in range(i_z + 1, k_z - 1):
            qc.cx(t, t + 1)
        if i_z + 1 != k_z:  # and j+1 != k and k-1 != j+1:
            qc.cx(k_z - 1, j_z + 1)
        # if j+1 != k:
        for t in range(j_z + 1, l_z - 1):
            qc.cx(t, t + 1)
        qc.cz(l_z, l_z - 1)
    elif i_z != k_z - 1:
        for t in range(i_z + 1, k_z - 1):
            qc.cx(t, t + 1)
        qc.cz(l_z, k_z - 1)
    qc.x(k)
    qc.x(i)

    qc.ry(theta / 8, l)
    qc.h(k)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.h(i)
    qc.cx(l, i)
    qc.ry(theta / 8, l)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.h(j)
    qc.cx(l, j)
    qc.ry(theta / 8, l)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.cx(l, i)
    qc.ry(theta / 8, l)
    qc.h(i)
    qc.cx(l, k)
    qc.ry(-theta / 8, l)
    qc.h(k)
    qc.cx(l, j)
    qc.h(j)

    qc.x(k)
    qc.x(i)
    if l_z != j_z + 1:
        qc.cz(l_z, l_z - 1)
        for t in range(l_z - 1, j_z + 1, -1):
            qc.cx(t - 1, t)
        if i_z + 1 != k_z:
            qc.cx(k_z - 1, j_z + 1)
        for t in range(k_z - 1, i_z + 1, -1):
            qc.cx(t - 1, t)
    elif i_z != k_z - 1:
        qc.cz(l_z, k_z - 1)
        for t in range(k_z - 1, i_z + 1, -1):
            qc.cx(t - 1, t)
    qc.cx(l, j)
    qc.cx(l, k)
    qc.cx(j, i)
    return qc
