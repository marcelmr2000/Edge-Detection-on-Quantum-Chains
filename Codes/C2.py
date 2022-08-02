"""
CONTENT:
    - Weighted Gram Matrix from Formula
    - SQRT Measurement
    - SDP (Programar-lo de nuevo)
"""

import numpy as np
import math
from scipy.linalg import sqrtm
import cvxpy as cp


N = 409

#List of all the possible J values given N and the projecting n

J_possibles = []
n1 = []

if N%2 == 0:
    for i in range(int(N/2)+1):
        J_possibles.append(i)
    n1.append(int(N/2))
else:
    for i in range(int(N/2)+1):
        J_possibles.append(1/2+i)
    n1.append(int(N/2-0.5))
    n1.append(int(N/2+0.5))

psq = []
psdp = []

#Document for the SDP and SRM results
SDP_doc = open("/Users/marcel/Desktop/TFG/Ps_SDP_SQRTM/SDP","a")
SQRTM_doc = open("/Users/marcel/Desktop/TFG/Ps_SDP_SQRTM/SQRTM","a")

for i in range(len(J_possibles)):
    
    #Create documents for G, Sqrt(G) and POVM elements
    Gram_Mat = open("/Users/marcel/Desktop/TFG/Matrices/Gram_Matrices/Gt,N={},J={}".format(N,J_possibles[i]),"w")
    Gram_Sqrt = open("/Users/marcel/Desktop/TFG/Matrices/GSQ/GSQ,N={},J={}".format(N,J_possibles[i]),"w")
    POVM = open("/Users/marcel/Desktop/TFG/Matrices/POVM_elements/POVM,N={},J={}".format(N,J_possibles[i]),"w")
    
    if i == 0:
        n_proj = n1
    else:
        if N%2 == 0:
            n_proj.insert(0,int(N/2)-(i))
            n_proj.append(int(N/2)+(i))
        else:
            n_proj.insert(0,int(N/2-1/2)-(i))
            n_proj.append(int(N/2+1/2)+(i))
            
    np.save("/Users/marcel/Desktop/TFG/Projecting_n/N={},J={}.npy".format(N,J_possibles[i]),n_proj)        
    
    #Creation of the Gram matrix
    G = np.zeros(shape = (len(n_proj),len(n_proj)))
    
    c = int(N/2-J_possibles[i])
    
    for j in range(len(n_proj)):
        n = n_proj[j]
        for k in range(len(n_proj)):
            nprim = n_proj[k]
            if n<=nprim:
                P1 = ((2*J_possibles[i]+1))/(np.sqrt((n+1)*(N-n+1)*(nprim+1)*(N-nprim+1)))
                P2 = np.sqrt((math.comb(n,c)*math.comb(N-nprim,c))/(math.comb(nprim,c)*math.comb(N-n,c)))
                G[j][k] = P1*P2
                G[k][j] = P1*P2
    
    #Sqrt(G)
    GSQ = sqrtm(G)
    if J_possibles[i] == J_possibles[-1]:
        
        GSQ1 = GSQ
        GSQ = np.zeros((len(n_proj), len(n_proj)))
        for p in range(len(n_proj)):
            for q in range(len(n_proj)):
                GSQ[p][q] = GSQ1[p][q].real
    np.save("/Users/marcel/Desktop/TFG/Matrices/GSQ_NPY/GSQ,N={},J={}.npy".format(N,J_possibles[i]),GSQ)
    
    #Export G and Sqrt(G)
    Gram_Mat.write("[")
    for i in range(len(n_proj)):
        Gram_Mat.write("[")
        for j in range(len(n_proj)):
            if j == len(n_proj)-1:
                Gram_Mat.write("{}".format(G[i,j]))
            else:
                Gram_Mat.write("{},".format(G[i,j]))
        Gram_Mat.write("]")
        if i != len(n_proj)-1:
            Gram_Mat.write(",\n")
    Gram_Mat.write("]")
    Gram_Mat.close()
    
    Gram_Sqrt.write("[")
    for i in range(len(n_proj)):
        Gram_Sqrt.write("[")
        for j in range(len(n_proj)):
            if j == len(n_proj)-1:
                Gram_Sqrt.write("{}".format(GSQ[i,j]))
            else:
                Gram_Sqrt.write("{},".format(GSQ[i,j]))
        Gram_Sqrt.write("]")
        if i != len(n_proj)-1:
            Gram_Sqrt.write(",\n")
    Gram_Sqrt.write("]")
    Gram_Sqrt.close()
    
    #SRM
    Sqrt_Measurement = 0

    for i in range(len(n_proj)):
        Sqrt_Measurement += np.round(GSQ[i,i]**(2),6)
        
    psq.append((1/N)*Sqrt_Measurement)
    
    omega_tilda = []
    for i in range(len(n_proj)):
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