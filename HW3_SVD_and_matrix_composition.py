# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:03:25 2020

@author: mtsco
"""
#%%
"""
Import needed modules
"""
import numpy as np
import math

#%%
"""
Define globals
"""
n = 2
k = 3
u = 0.5
#%%
"""
Contruct the matrix
"""
A = np.zeros(shape=(n,n))

for row in range (0, n):
    for col in range(0, n):
        #rules for which values should be filled in
        if col <= row < (col + k) or ((col + k) > 6 and row < (col+k)%n) :
            A[row, col] = 1 - u
        else:
            A[row, col] = 0 - u
print(A)
#np.matrix.transpose(A)
#%%

def euclidean_normalized(vt):
    for j in range(0, vt.shape[1]):
        s = 0
        for i in range(0, vt.shape[0]):
            s = s + float(vt[i, j])*float(vt[i, j])
        norm = math.sqrt(np.around(s, 8))
        for i in range(0, vt.shape[0]):
            vt[i,j] = vt[i,j]/norm
    return vt
        
            
def round_all_vals(mat, x):
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            mat[i, j] = np.around(mat[i, j], x)
    return mat

def SVD(A):
    #get At*A
    a_t_a = np.dot(np.matrix.transpose(A), A)
    
    #get the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(a_t_a)   
    
    vals = vals[::-1]
    
    #sorting the vec array
    for i in range(0, int(vecs.shape[0]/2)):
        temp = []
        #this is stupid but apparently python assigns all arrays by reference not value so temp will change
        for j in range(0, vecs.shape[1]):
            temp.append(vecs[i,j])
        vecs[i,:] = vecs[vecs.shape[0]-i -1, :]
        vecs[vecs.shape[0]-i -1, :] = temp
    
    #this gets rid of an errors from calculating the eigenvalues as very small negative numbers
    vals = np.around(vals, 8)

    #computing sigma
    sigma = np.zeros((n, n))
    V = np.zeros((n,n))
    for j in range (0, n):
        sigma[j, j] = math.sqrt(vals[j])
        V[:,j] = vecs[j]

    sigV = np.matrix.dot(sigma, V)
    AsigV = np.matrix.dot(A, sigV)
    U = euclidean_normalized(AsigV)
    Vt = np.matrix.transpose(V)
    
    return U, sigma, Vt

#%%
U, sigma, Vt = SVD(A)
sigV = np.matrix.dot(sigma, Vt)
UsigV = np.matrix.dot(U, sigV)
print(A)
print(round_all_vals(UsigV, 2))



#%%  

matA = np.zeros((2, 2))
matA[0,0] = 2
matA[0,1] = 1
matA[1,0] = 4
matA[1,1] = 2

#%%
matU = np.zeros((2,2))
matU[0,0] = (-3.0/(5*math.sqrt(5)))
matU[1,0] = (-6.0/(5*math.sqrt(5)))
matU[0,1] = (4.0/math.sqrt(5))
matU[1,1] = (6.0/math.sqrt(5))

matSig = np.zeros((2,2))
matSig[0,0] = 5.0

print(matU)
#%%
matVt = np.zeros((2,2))
matVt[0,0] = (2.0/math.sqrt(5))
matVt[0,1] = (-1.0/math.sqrt(5))
matVt[1,0] = (1.0/math.sqrt(5))
matVt[1,1] = (2.0/math.sqrt(5))

print(round_all_vals(matVt, 2))
#%%
SigVt = np.dot(matSig, matVt)


#print(matVt)


USigVt = np.dot(matU, SigVt)

AV = np.dot(matA, SigVt)

print("this is AV correct")
print(AV)

U = euclidean_normalized(AV)


SigV = np.matrix.dot(matSig, np.matrix.transpose(matVt))
USigV = np.matrix.dot(U, SigV)

print(USigV)

A_calc = round_all_vals(USigV, 0)
print(A_calc)

#print(np.dot((1/5)*matA, matVt[:,0]))

#print(np.dot(matA, matVt[:,1]))





#%%
U1, S1, Vt1 = SVD(matA)

#print(U1)
#print(S1)
#print(Vt1)

sigV1 = np.matrix.dot(S1, Vt1)
UsigV1 = np.matrix.dot(U1, sigV1)
print(round_all_vals(UsigV1, 3))
#
print(matA)




#%%
