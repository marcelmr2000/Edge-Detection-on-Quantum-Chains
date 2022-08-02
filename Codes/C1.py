"""
CONTENT:

- Omega Vectors
- Weighted Gram Matrix
- SQRT Measurement
- SDP
"""

import numpy as np
import sympy as smp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sympy.physics.quantum.cg import CG
from sympy.utilities.iterables import multiset_permutations
import cvxpy as cp

N = 11

#List of all the possible J values given N

J_possibles = []

if N%2 == 0:
    for i in range(int(N/2)+1):
        J_possibles.append(i)
else:
    for i in range(int(N/2)+1):
        J_possibles.append(1/2+i)

#Function that calculates the Omega Projectors

def Omega_Vector(N,n,J):
    s = []
    Alphas = []

    for i in range(n):
        s.append(smp.Rational(1,2))
    for i in range(N-n):
        s.append(smp.Rational(-1,2))
        
    w = np.cumsum(s)

    J_max = N/2
    Steps_down = int(J_max - J)

    A = []
    for i in range(Steps_down):
        A.append(smp.Rational(-1,2))
    for i in range(N-Steps_down):
        A.append(smp.Rational(1,2))


    S = list(multiset_permutations(A))
    for i in S:
        if ((all(x >= 0 for x in np.cumsum(i)) == True) and (all(y>=0 for y in np.cumsum(i)-w))):
            Alphas.append(np.cumsum(i))

    Omega = []

    for Alfa in Alphas:
        Prod = 1
        for i in range(N-1):
            Prod = Prod*CG(Alfa[i],w[i],smp.Rational(1,2),w[i+1]-w[i],Alfa[i+1],w[i+1]).doit()
        Omega.append(Prod)


    Norm = sum(i*i for i in Omega)
    if Norm == 0:
        Norm = 1
    
    Omega_Norm = []
    for i in Omega:
        Omega_Norm.append(i/smp.sqrt(Norm))
       
    return Omega_Norm[::-1]


#Array for the probabilities both on SRM and SDP

psq=[]
psdp = []

#Documents where the success probability for N will be exported for SRM & SDP
SDP_doc = open("/Users/marcel/Desktop/TFG/Ps_SDP_SQRTM/SDP","a")
SQRTM_doc = open("/Users/marcel/Desktop/TFG/Ps_SDP_SQRTM/SQRTM","a")

for j in range(len(J_possibles)):
    
    #Documents where the Gram matrices, the Sqrt(G) and the POVM elements of the SDP will be exported
    Gram_Sqrt = open("/Users/marcel/Desktop/TFG//Matrices/GSQ/GSQ,N={},J={}".format(N,J_possibles[j]),"w")
    Gram_Mat = open("/Users/marcel/Desktop/TFG/Matrices/Gram_Matrices/Gt,N={},J={}".format(N,J_possibles[j]),"w")
    POVM = open("/Users/marcel/Desktop/TFG/Matrices/POVM_elements/POVM,N={},J={}".format(N,J_possibles[j]),"w")
    
    
    Omegas = []
    n1 = []
    
    for n in range(N+1):
        if all(x == 0 for x in Omega_Vector(N,n,j)) == True:
            continue
        else:
            Omegas.append(Omega_Vector(N,n,j))
            n1.append(n)
    
    #Export an array with all the hypotheses that project on angular momentum j
    np.save("/Users/marcel/Desktop/TFG/Projecting_n/N={},J={}.npy".format(N,J_possibles[j]),n1)
    
    #Define an empty G matrix
    G = smp.zeros(len(Omegas),len(Omegas))
    
    #Fill the G matrix

    for i in range(len(Omegas)):
        for m in range(len(Omegas)):        
            if len(Omegas[i])==len(Omegas[m]):
                for k in range(len(Omegas[i])):
                    G[i,m] = G[i,m] + Omegas[i][k]*Omegas[m][k]
                    
            elif len(Omegas[i])<len(Omegas[m]):
                for k in range(len(Omegas[i])):
                    G[i,m] = G[i,m] + Omegas[i][k]*Omegas[m][k]
            
            else:
                for k in range(len(Omegas[m])):
                    G[i,m] = G[i,m] + Omegas[i][k]*Omegas[m][k]
    
    #Weight the G matrix
    Di = smp.zeros(len(Omegas),len(Omegas))
    for i in range(len(Omegas)):
        Di[i,i] = ((np.sqrt(2*J_possibles[j]+1))/(np.sqrt(n1[i]+1)*np.sqrt(N-n1[i]+1)))
    
    #Gt is the weighted Gram matrix
    Gt = (Di*G*Di).evalf()
    
    #Sqrt(G)
    GSQ = sqrtm(np.array(Gt).astype(np.float64))
    
    if J_possibles[j] == J_possibles[-1]:
        GSQ1 = GSQ
        GSQ = np.zeros((len(Omegas), len(Omegas)))
        for i in range(len(Omegas)):
            for k in range(len(Omegas)):
                GSQ[i][k] = GSQ1[i][k].real
    
    #Save Sqrt(G) as an array        
    np.save("/Users/marcel/Desktop/TFG/Matrices/GSQ_NPY/GSQ,N={},J={}.npy".format(N,J_possibles[j]),GSQ)
    
    #Export G, and Sqrt(G) in matrix form
    Gram_Mat.write("[")
    for i in range(len(Omegas)):
        Gram_Mat.write("[")
        for j in range(len(Omegas)):
            if j == len(Omegas)-1:
                Gram_Mat.write("{}".format(Gt[i,j]))
            else:
                Gram_Mat.write("{},".format(Gt[i,j]))
        Gram_Mat.write("]")
        if i != len(Omegas)-1:
            Gram_Mat.write(",")
    Gram_Mat.write("]")
    Gram_Mat.close()
    
    Gram_Sqrt.write("[")
    for i in range(len(Omegas)):
        Gram_Sqrt.write("[")
        for j in range(len(Omegas)):
            if j == len(Omegas)-1:
                Gram_Sqrt.write("{}".format(GSQ[i,j]))
            else:
                Gram_Sqrt.write("{},".format(GSQ[i,j]))
        Gram_Sqrt.write("]")
        if i != len(Omegas)-1:
            Gram_Sqrt.write(",")
    Gram_Sqrt.write("]")
    Gram_Sqrt.close()
    
    #SRM
    Sqrt_Measurement = 0

    for i in range(len(Omegas)):
        Sqrt_Measurement += np.round(GSQ[i,i]**(2),6)
        
    psq.append((1/N)*Sqrt_Measurement)
    
    
    #SDP
    omega_tilda = []
    for i in range(len(Omegas)):
        omega_tilda.append(GSQ[:,i])

    #Definition of variables and constraints of SDP
    X = []
    for i in range(len(omega_tilda)):
        X.append(cp.Variable((len(omega_tilda[0]),len(omega_tilda[0])),PSD=True))

    constraints = [sum(X) == np.identity(len(omega_tilda[0]))]
    
    #Define the problem and Solve it
    P = 0
    for i in range(len(omega_tilda)):
        P += (1/N)*omega_tilda[i]@ X[i]@ omega_tilda[i]
    
    Prob = cp.Problem(cp.Maximize(P),constraints)
    Prob.solve()

    psdp.append(Prob.value)
    
    #Export the POVM elements in matrix form
    for i in range(len(X)):
        POVM.write("[")
        for l in range(len(X[i].value)):
            POVM.write("[")
            for j in range(len(X[i].value)):
                if j == len(X[i].value)-1:
                    POVM.write("{}".format(X[i].value[l,j]))
                else:
                    POVM.write("{},".format(X[i].value[l,j]))
            POVM.write("]")
            if l != len(X[i].value)-1:
                POVM.write(",")
        POVM.write("]")
        POVM.write("\n\n")
    POVM.close()
    
psq.pop(-1)
psq.append(1/N)

#Append the SDP and SRM documents with the success probability of N
SDP_doc.write("{}\t{}\n".format(N,np.cumsum(psdp)[-1]))
SQRTM_doc.write("{}\t{}\n".format(N,np.cumsum(psq)[-1]))
SDP_doc.close()
SQRTM_doc.close()

#Documents for the success probabilities of j given N
SDP_partial = open("/Users/marcel/Desktop/TFG/Ps_SDP_SQRTM/Parcials/SDP_N={}".format(N),"w")
SQRT_partial = open("/Users/marcel/Desktop/TFG/Ps_SDP_SQRTM/Parcials/SQRT_N={}".format(N),"w")

SDP_partial.write("J\tSDP\n")
SQRT_partial.write("J\tSQRTM\n")

for i in range(len(psq)):
    SQRT_partial.write("{}\t{}\n".format(J_possibles[i], psq[i]))
    SDP_partial.write("{}\t{}\n".format(J_possibles[i], psdp[i]))

SQRT_partial.close()
SDP_partial.close()
