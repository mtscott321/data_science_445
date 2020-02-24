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
n = 3
k = 3
u = 0
#%%
"""
Construct the matrix
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
def multiply_three(A, B, C):
    BC = np.matrix.dot(B, C)
    ABC = np.matrix.dot(A, BC)
    return ABC


def euclidean_normalized(vt):
    for j in range(0, vt.shape[1]):
        s = 0
        for i in range(0, vt.shape[0]):
            s = s + float(vt[i, j])*float(vt[i, j])
        norm = math.sqrt(np.around(s, 8))
        for i in range(0, vt.shape[0]):
            if norm != 0:
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
    
    print("\nthese are the eignenvecs")
    print(round_all_vals(vecs, 2))
    #vals = vals[::-1]
    
    #this gets rid of any errors from calculating the eigenvalues as very small negative numbers
    vals = np.around(vals, 8)

    #computing sigma and V
    sigma = np.zeros((n, n))
    V = np.zeros((n,n))
    for j in range (0, n):
        #vals is returned in the reverse order
        #this assigns the diagonal of sigma as the sqrt of the eigenvals
        sigma[j, j] = math.sqrt(vals[vals.shape[0] - 1 -j])
        #vecs is returned in the reverse order
        #this assigns the columns of V as the eigenvectors
        V[j,:] = vecs[j][::-1]
    Ut = np.zeros((n,n))
    #computing U
    for j in range(0, n):
        if sigma[j,j] != 0:
            Ut[j,:] = (1/sigma[j,j])*np.matrix.dot(A, V[:,j])

    U_norm = euclidean_normalized(np.matrix.transpose(Ut))
    Vt = np.matrix.transpose(V)
    
    return U_norm, sigma, Vt

#%%
test = np.array([[4, 2, 0],
                [7, 6, 9],
                [6, 6, 6]])
print(test)

A, B, C = SVD(test)
print("\nThis is U")
print(A)
print("\nThis is Sigma")
print(B)
print("\nThis is Vt")
print(C)
print("\nThis is their product")
print(round_all_vals(multiply_three(A, B, C), 0))

#%%

test = np.array([[4, 2, 0],
                [7, 6, 9],
                [6, 6, 6]])
print(test)

A, B, C = SVD(test)

Ct = np.matrix.transpose(C)
U = np.zeros((n,n))
for j in range(0, n):
    print(Ct[:,j])
    
    print((1/B[j,j])*np.matrix.dot(test, Ct[:,j]))




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
U1, S1, Vt1 = SVD(A)

print(round_all_vals(U1, 2))

print(round_all_vals(S1, 2))
print(round_all_vals(Vt1, 2))

sigV1 = np.matrix.dot(S1, Vt1)
UsigV1 = np.matrix.dot(U1, sigV1)
#print(round_all_vals(UsigV1, 2))
#
print(A)
