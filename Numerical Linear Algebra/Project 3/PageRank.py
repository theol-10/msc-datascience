import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import time


#Function for creating sparse matrix D
def create_D(G):

    n_j = np.sum(G,axis=0)   #out degree of page 
    d = np.zeros(G.shape[0])  #diagonal of desired matrix

    for i in range(0,G.shape[0]):
        if n_j[0,i] == 0: d[i] = 0
        else: d[i]=1/n_j[0,i]

    return sp.diags(d)   # return D as a sparse matrix 


#Function for creating sparse matrix A=GD
def create_A(D,G):

    A = G.dot(D)
    return A


#Function to compute the PR vector of M_m using the power method (with storing matrices)
def PR_store(A,tol,m):

    n = A.shape[0]
    e = np.ones(n)
    z = np.ones(n)/n
    z[np.unique(A.indices)] = m/n  #A.indices to get the column position of the non zero values
    x_0 = np.zeros(n)
    x_k = np.ones(n) / n 
    
    while np.linalg.norm(x_0-x_k,np.inf)>tol:
        x_0 = x_k
        x_k = (1-m)*A.dot(x_0) + e*(np.dot(z,x_0))

    x_k = x_k / np.sum(x_k)  #Normalization of vector
    return x_k


#Function to computes the PR vector of M_m using the power method (without storing matrices)
def PR_without_store(G,tol,m):

    n = G.shape[0]
    L = []
    n_j = []

    for j in range(0,n):
        L_j = G.indices[G.indptr[j]:G.indptr[j+1]]   #webpages with link to page j
        L.append(L_j)
        n_j.append(len(L_j))
    
    x = np.zeros(n)
    xc = np.ones(n) / n 

    while np.linalg.norm(x-xc,np.inf)>tol:
        xc = x
        x = np.zeros(n)
        for j in range (0, n):
            if(n_j[j] == 0):
                x = x + xc[j] / n
            else:
                for i in L[j]:
                    x[i] = x[i] + xc[j] / n_j[j]
        x = (1 - m) * x + m / n

    x = x / np.sum(x) #Normalization of vector
    return x
        

if __name__ == '__main__':
    
    G = mmread('p2p-Gnutella30.mtx')
    D = create_D(G)
    A = create_A(D,G)


    st1 = time.time()
    x1 = PR_store(A,1e-12,0.15)  #computing the PR vector of M_m using the power method (storing matrices)
    et1 = time.time()
    print('Solution of PR vector of M_m using the power method (storing matrices):')
    print(x1)
    print('Computational time:', et1-st1,'seconds')
    print('\n')

    st2 = time.time()
    x2 = PR_without_store(sp.csc_matrix(G),1e-12,0.15)  #computing the PR vector of M_m using the power method (without storing matrices)
    et2 = time.time()
    print('Solution of PR vector of M_m using the power method (without storing matrices):')
    print(x2)
    print('Computational time:', et2-st2,'seconds')
    print('\n')    
    print('Difference between the solutions',np.linalg.norm(x1-x2))
    