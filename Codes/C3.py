"""
Hamming Distance expected value SRM:
    - Creates a document with the Expected Value for each J
    - Document with the total Expected value for each N
"""

import numpy as np
import cvxpy as cp

N = 40

#List of all the possible J values given N
J_possibles = []

if N%2 == 0:
    for i in range(int(N/2)+1):
        J_possibles.append(i)
else:
    for i in range(int(N/2)+1):
        J_possibles.append(1/2+i)

#Document where <h> will be exported for each N
g = open("/Users/marcel/Desktop/TFG/HammingDistance_Optimization/ExpectedValueSRM","a") 

Ham_Opt = 0

for J in J_possibles:
    #Load Sqrt(G) and the corresponding n array of projections
    GSQ = np.load("/Users/marcel/Desktop/TFG/Matrices/GSQ_NPY/GSQ,N={},J={}.npy".format(N,J))
    n_proj = np.load("/Users/marcel/Desktop/TFG/Projecting_n/N={},J={}.npy".format(N,J))
    
    #Perform <h>
    for i in range(len(n_proj)):
        for j in range(len(n_proj)):
            Ham_Opt += (1/N)*abs(i-j)*GSQ[i][j]**2

g.write("{}\t{}\n".format(N,Ham_Opt))
g.close()
