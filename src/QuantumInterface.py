from pyscf import gto, scf 
import numpy as np
from functools import *
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.quantum_info import Operator, Pauli, process_fidelity
from qiskit.circuit import Parameter
from qiskit_aer.backends import AerSimulator
from qiskit import transpile

import math as mt
import scipy
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp

atm = '''
C       0.000    0.5952006020    0.0000069804;
C       0.000   -0.5952006020    0.0000069804;
H       0.000    1.6607383043   -0.0000069804;
H       0.000   -1.6607383043   -0.0000069804;
'''

mol = gto.M(
    atom=atm,
    #charge= 2,
    charge= 0,
    spin = 0,
    verbose=2,
    unit="Angstrom",
    basis= "def2-svp"
)

mf = scf.RHF(mol)
print(mol.nelec)
mf.max_cycle = 300
mf.kernel()

print(mol.nelec)
print(mf.e_tot - mf.energy_nuc())

c_mo = mf.mo_coeff
h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
g_ao = mol.intor("int2e")


mo_occa = (mf.mo_occ > 0).astype(np.double)
mo_occb = (mf.mo_occ == 2).astype(np.double)
print(mo_occa)
print(mo_occb)
ncas = 2
nclosed = sum(mol.nelec)//2 - ncas//2
print(nclosed)

nvirt = len(mo_occa) - (nclosed + ncas)
core0_coeff = mf.mo_coeff[:, :nclosed]
active0_coeff = mf.mo_coeff[:, nclosed: nclosed + ncas]
virtual0_coeff = mf.mo_coeff[:, nclosed + ncas:]

print(mo_occa.sum())

mon_occ = int(mo_occa.sum())
occ_coeff = mf.mo_coeff[:,:mon_occ]
print(occ_coeff.shape)
mondm = occ_coeff@occ_coeff.T*2
print(mondm.shape)
atm2 = '''C       0.000    0.5952006020   4.0000069804;
C       0.000   -0.5952006020   4.0000069804;
H       0.000    1.6607383043   4.0000069804;
H       0.000   -1.6607383043   4.0000069804;
'''

mol = gto.M(
    atom=atm2,
    #charge= 2,
    charge= 0,
    spin = 0,
    verbose=3,
    unit="Angstrom",
    basis= "def2-svp"
)

mf = scf.RHF(mol)
print(mol.nelec)
mf.max_cycle = 300
mf.kernel(mondm)



core1_coeff = mf.mo_coeff[:, :nclosed]
active1_coeff = mf.mo_coeff[:, nclosed: nclosed + ncas]
virtual1_coeff = mf.mo_coeff[:, nclosed + ncas:]

dim_coeff = np.zeros((int(mondm.shape[0]*2), int(mon_occ*2)))
dim_coeff[:int(mondm.shape[0]),:int(mon_occ)] = occ_coeff
dim_coeff[int(mondm.shape[0]):int(2*mondm.shape[0]),mon_occ:int(2*mon_occ)] = occ_coeff  
dimdm = dim_coeff@dim_coeff.T*2

print(dimdm.shape)
print(atm+atm2)

dimer_mol = gto.M(
    atom=atm+atm2,
    #charge= 2,
    charge= 0,
    spin = 0,
    verbose=3,
    unit="Angstrom",
    basis= "def2-svp"
)
dimer_mf = scf.RHF(dimer_mol)
print(core1_coeff.shape)
dim_core_coeff = np.zeros((int(mondm.shape[0]*2), int(core1_coeff.shape[1]*2)))
print(dim_core_coeff.shape)

dim_core_coeff[ : core0_coeff.shape[0] , : core0_coeff.shape[1] ] = core0_coeff 
dim_core_coeff[ core0_coeff.shape[0] : core0_coeff.shape[0] + core1_coeff.shape[0] , core0_coeff.shape[1] :  ] = core1_coeff 

dim_active_coeff = np.zeros((int(mondm.shape[0]*2), int(active1_coeff.shape[1]*2)))
dim_active_coeff[ : active0_coeff.shape[0] , : active0_coeff.shape[1] ] = active0_coeff
dim_active_coeff[ active0_coeff.shape[0] : active0_coeff.shape[0] + active1_coeff.shape[0] , active0_coeff.shape[1] : active0_coeff.shape[1]+ active1_coeff.shape[1] ] = active1_coeff 


hcore = dimer_mol.intor("int1e_kin") + dimer_mol.intor("int1e_nuc")
eri_ao = dimer_mol.intor("int2e_sph")

dim_core_dm = dim_core_coeff@dim_core_coeff.T
print(dim_core_dm.shape)


vj_core_1 = np.einsum("ijkl,kl-> ij", eri_ao, dim_core_dm)
vk_core_1 = np.einsum("ikjl,kl-> ij", eri_ao, dim_core_dm)
dim_corevhf = vj_core_1 + vj_core_1 - vk_core_1

energy_core = 0# mol.energy_nuc()
energy_core += np.einsum('ij,ji', dim_core_dm, hcore)
energy_core += np.einsum('ij,ji', dim_core_dm, hcore)
energy_core += np.einsum('ij,ji', dim_core_dm, dim_corevhf)
#energy_core += np.einsum('ij,ji', dim_core_dm, dim_corevhf)


h1eff = dim_active_coeff.T@(hcore + dim_corevhf)@dim_active_coeff

eris1 = np.einsum("ip,jq,ijkl,kr,ls->pqrs", dim_active_coeff, dim_active_coeff, 
                                 eri_ao, dim_active_coeff, dim_active_coeff)

print(h1eff)
print(eris1)
overlap = dim_active_coeff.T@dim_active_coeff

S1a_ = "SIIIIIII"
S1b_ = "ZSIIIIII"
S2a_ = "ZZSIIIII"
S2b_ = "ZZZSIIII"
S3a_ = "ZZZZSIII"
S3b_ = "ZZZZZSII"
S4a_ = "ZZZZZZSI"
S4b_ = "ZZZZZZZS"


#S1a_ = "SZZZZZZZ"
#S1b_ = "ISZZZZZZ"
#S2a_ = "IISZZZZZ"
#S2b_ = "IIISZZZZ"
#S3a_ = "IIIISZZZ"
#S3b_ = "IIIIISZZ"
#S4a_ = "IIIIIISZ"
#S4b_ = "IIIIIIIS"

g = [S1a_, S1b_, S2a_, S2b_ ,S3a_, S3b_, S4a_, S4b_]

mat = []
for i in range(len(g)):
    ind = g[i].index("S")
    sn = list(g[i])
    sn[i] = 'X'
    P1 = "".join(sn)
    sn[i] = 'Y'
    P2 = "".join(sn)
    mat.append(SparsePauliOp([P1,P2],np.array([0.5,-0.5j])))


print(mat)

def clean_paulis(gp):
    spl1 = sum(gp)
    sl = list(spl1.paulis)
    sl2 = list(spl1.coeffs)

    idx = 0
    while(True):
        idxs = []
        for i in range(len(sl)):
            if(sl[i] == sl[idx]):
                idxs.append(i)
        #print(len(sl))
        #print(idxs)
        for i in range(len(idxs)-1,0, -1):
            #print(i)
            #print(idxs[i])
            sl.pop(idxs[i])
            sl2[idxs[0]] += sl2[idxs[i]]
            sl2.pop(idxs[i])
        #print(len(sl))
        if(len(sl)-1 == idx):
            break
        else:
            idx += 1


    for i in range(len(sl2)-1, 0, -1):
        if(abs(sl2[i]) < 1e-10):
            sl.pop(i)
            sl2.pop(i)

    #print(sl2)
    #print(sl)
    bl = SparsePauliOp(sl,sl2)
    return bl



gp = []
for i in range(len(mat)):
    for j in range(len(mat)):
        if((i%2) == (j%2)):
            gp.append(mat[i]@mat[j].adjoint()*overlap[i//2,j//2])

ol = clean_paulis(gp)
print(ol)
#print(bl.to_matrix())
olm =ol.to_matrix()


                
gp = []
for i in range(len(mat)):
    for j in range(len(mat)):
        if((i%2) == (j%2)):
               gp.append(mat[i]@mat[j].adjoint()*h1eff[i//2,j//2])


bl = clean_paulis(gp)
print(bl)
#print(bl.to_matrix())
blm =bl.to_matrix()




EIP = []
for i in range(len(mat)):
    for j in range(len(mat)):
        for k in range(len(mat)):
            if((j%2) == (k%2)):
                for l in range(len(mat)):
                    if((i%2) == (l%2)):
                        EIP.append(mat[i]@mat[j]@mat[k].adjoint()@mat[l].adjoint()*eris1[i//2,l//2,j//2,k//2]) # pysicst to chemist notaion
                        # read sabo ausland
                
#print(EIP)


kl = clean_paulis(EIP)
print(kl)

fullkl = bl + 0.5*kl
fullkl = clean_paulis(fullkl)
print(fullkl)
#print(kl.to_matrix())
klm = kl.to_matrix()

alm = fullkl.to_matrix()


ag0 = np.array([[1.0,0.0]])
ag1 = np.array([[0.0,1.0]])
#adg = reduce(np.kron, [ag0,ag0, ag0, ag0,ag0,ag0, ag0, ag0])
#mlp  = S1b@S1a@adg.T
#mlp2  = S1a@S1b@adg.T
#print(mlp)
#   monomer1           monomer2
#         |                   |
#         V                   V

#     [ I , I , 0 , 0 , I , I , 0 , 0 ]

adg1 = reduce(np.kron, [ag1,ag1, ag0, ag0,ag1,ag1, ag0, ag0])
print(adg1.T)
print(alm.shape)
print(adg1.shape)
nk = adg1@alm@adg1.T
print(alm@adg1.T)
print(energy_core)
print(nk)


nk2 = adg1@alm@adg1.T

print(nk2)
print(adg1@adg1.T)
print(h1eff)

print("Printing classical energy")
print(energy_core + nk2)
print(energy_core + nk2 + dimer_mf.energy_nuc())
dimer_mf.kernel()
print(dimer_mf.e_tot - dimer_mf.energy_nuc())
print(mf.e_tot*2)

import ts7
import exC1
sl = list(fullkl.paulis)
sl2 = list(fullkl.coeffs)
mat_list = []
for i in range(len(sl)):
    #print(str(sl[i]))
    #print(sl2[i])
    mat_list.append({"mat": str(sl[i]), "coef":sl2[i] })


theta = [ Parameter("r"+str(i)) for i in range(10) ] 
theta_len = 0
class MyFucntion(ts7.Function):
    def __init__(self, nqubits, mat_lista):
        super().__init__(nqubits, mat_lista)
        # additional initialization if needed
    def add_ansatz(self,nansatz):
        #state preparation
        self.qc.x(2)
        self.qc.x(3)
        self.qc.x(6)
        self.qc.x(7)
        
        #anzatz addition
        l = 0
        for p in range(0,2*ncas -1,2 ):
            exC1.single_excitation_efficient(2*p+2,2*p,8,self.qc, theta[l],)
            l += 1
            exC1.single_excitation_efficient(2*p+1,2*p-1,8,self.qc, theta[l],)
            l += 1
            exC1.double_excitation_efficient(2 * p + 3, 2 * p + 2, 2 * p, 2 * p + 1,8,self.qc, theta[l],)
            l += 1
        #exC1.double_excitation_efficient(5,2,7,4,8,self.qc, theta3,)
        #exC1.double_excitation_efficient(5,2,7,5,8,self.qc, theta3,)
        theta_len = l
        self.grad_parametrs = [2 for i in range(l)]
        #self.qc.x(0)
        #self.qc.x(1)
        #self.qc.x(4)
        #self.qc.x(5)
        
        print(self.qc)
        return theta_len
        
      
func = MyFucntion(8, mat_list)
theta_len = func.add_ansatz(8)
func.prepare_circuit()
func.evaluate([0.0 for i in range(theta_len)] )
energy = func.Energy()
print("Energy: ", energy, energy_core+ energy)
print(energy)
print(energy_core)
print(energy_core + energy)
print("quantum Comp",energy_core + energy + dimer_mf.energy_nuc())
print(energy_core + nk2 + dimer_mf.energy_nuc())
print(dimer_mf.e_tot - dimer_mf.energy_nuc())
print(mf.e_tot*2)

def f(x):
    func.evaluate(x)
    return func.Energy()


# Gradient of the function
def grad(x):
    return np.array(func.Gradient(x))

# Print function for each iteration
def callback(x):
    print(f"Current x: {x}, f(x) = {f(x)}")

# Initial guess
# x0 = np.array([0. for i in range(theta_len)])
x0 = np.array([ 0.00703305+0.j, -0.00175921+0.j, -0.1343368 +0.j, -0.00274689+0.j, 0.00913444+0.j, -0.12924041+0.j])
print(x0)
print(f(x0))
#exit()
# Run BFGS
t = x0.copy()
para = {}

for i in range(len(func.qc.parameters)):
    para[func.qc.parameters[i]] = np.array([t[i]])

print(para)
qc_bound = func.qc.assign_parameters(para)
g = Operator(qc_bound)
mat = g.to_matrix()
print(mat)
adg1 = reduce(np.kron, [ag0,ag0, ag0, ag0,ag0,ag0, ag0, ag0])
print(mat@adg1.T)

exit()
result = minimize(
    fun=f,
    x0=x0,
    method="BFGS",
    jac=grad,      # Provide gradient
    callback=callback,
    options={"disp": True}
)
print(result.x)
t = result.x.copy()
para = {}

for i in range(len(func.qc.parameters)):
    para[func.qc.parameters[i]] = np.array([t[i]])

print(para)
qc_bound = func.qc.assign_parameters(para)
g = Operator(qc_bound)
mat = g.to_matrix()
adg1 = reduce(np.kron, [ag0,ag0, ag0, ag0,ag0,ag0, ag0, ag0])
print(mat)
print(mat@adg1.T)
