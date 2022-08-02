"""
Hamming Distance Optimization:
    - Creates a document with the Expected Optimized Value of <h> for each J
    - Document with the total optimized expected value <h> for each N
"""

import numpy as np
import cvxpy as cp

N = 7

#List of all the possible J values given N
J_possibles = []

if N%2 == 0:
    for i in range(int(N/2)+1):
        J_possibles.append(i)
else:
    for i in range(int(N/2)+1):
        J_possibles.append(1/2+i)


Ham_Opt = 0

#Create document for <h> for each j given N and another document for <h> given N
f = open("/Users/marcel/Desktop/TFG/HammingDistance_Optimization/Partials/HammingOpt_N,N={}".format(N),"a")
g = open("/Users/marcel/Desktop/TFG/HammingDistance_Optimization/ExpectedValue_N","a")

f.write("J\tHOpt/N\n")

for J in J_possibles:
    
    #Load Sqrt(G) and the corresponding n array of projections
    GSQ = np.load("/Users/marcel/Desktop/TFG/Matrices/GSQ_NPY/GSQ,N={},J={}.npy".format(N,J))
    n_proj = np.load("/Users/marcel/Desktop/TFG/Projecting_n/N={},J={}.npy".format(N,J))
    
    #Get the columns of Sqrt(G) in order to create density matrices |a><a|
    GSQ_col = []

    for i in range(len(GSQ[0])):
        GSQ_col.append(GSQ[:][i])
        
    #Define Variables and Constrains of SDP
    X = []
    for i in range(len(GSQ_col)):
        X.append(cp.Variable((len(GSQ_col[0]),len(GSQ_col[0])),PSD=True))

    constraints = [sum(X) == np.identity(len(GSQ_col[0]))]
    
    #Define and Solve the Problem
    P = 0 
    for i in range(len(n_proj)):
        for j in range(len(n_proj)):
            P += (1/N)*abs(i-j)*(GSQ_col[i]@X[j]@GSQ_col[i])
            
    Prob = cp.Problem(cp.Minimize(P),constraints)
    Prob.solve()

    Ham_Opt += Prob.value
    
    #Export <h> for each j
    f.write("{}\t{}\n".format(J,Prob.value))

#Export <h> for N
g.write("{}\t{}\n".format(N,Ham_Opt))

f.close()
g.close()