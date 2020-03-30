# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:33:54 2020

@author: mtsco
"""
import math
import numpy as np

def round_all_vals(mat, x=2):
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            mat[i, j] = np.around(mat[i, j], x)
    return mat

def multiply_three(A, B, C):
    BC = np.matrix.dot(B, C)
    ABC = np.matrix.dot(A, BC)
    del BC
    return ABC

def euclidean_normalize_cols(A):
    for j in range(0, A.shape[1]):
        s = 0
        for i in range(0, A.shape[0]):
            s = s + float(A[i, j])*float(A[i, j])
        norm = math.sqrt(np.around(s, 8))
        for i in range(0, A.shape[0]):
            if norm != 0:
                A[i,j] = A[i,j]/norm
    return A

def SVD(A):
    [D, n] = A.shape
    min_Dn = D
    if D > n:
        min_Dn = n
    #get At*A
    a_t_a = np.dot(np.matrix.transpose(A), A)
    
    #get the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(a_t_a)   

    #this gets rid of any errors from calculating the eigenvalues as very small negative numbers
    vals = np.around(vals, 8)

    #computing sigma and V
    sigma = np.zeros((D, n))
    V_temp = np.zeros((n,n))
    
    #making sigma
    for j in range (0, min_Dn):
        #vals is returned in the reverse order
        #this assigns the diagonal of sigma as the sqrt of the eigenvals
        sigma[j, j] = math.sqrt(vals[vals.shape[0] - 1 -j])
        
    #making V
    for i in range(0, n):
        #vecs is returned in the reverse order
        #this assigns the columns of V as the eigenvectors
        temporary = vecs[i]
        V_temp[i,:] = temporary

    V = np.fliplr(V_temp)
    Ut = np.zeros((D, D))
    
    #computing U
    for j in range(0, min_Dn):
        if sigma[j,j] != 0:
            Ut[j,:] = (1/sigma[j,j])*np.matrix.dot(A, V[:,j])

    U_norm = euclidean_normalize_cols(np.matrix.transpose(Ut))
    Vt = np.matrix.transpose(V)
    
    return U_norm, sigma, Vt

"""
returns a matrix of values, each ith column (out of n) chosen from a normal 
distribution with mean mus[i] and standard deviation sig[i]
"""

def normal_dist_mat(D, n, sig, mus):
    A = np.zeros((D, n))
    for col in range(0, n):
        new_col = np.random.normal(loc = mus[col], scale = sig[col], size = (D))
        A[:, col] = new_col
    return A

def normal_stnd_dist_mat(D, n):
    return normal_dist_mat(D, n, sig = np.ones([n]), mus = np.zeros([n]))

"""
n many times, picks a cluster from the prob.size posibilities, with each ith 
cluster having a probability of prob[i] of being picked. Returns an array of the 
selected clusters (starting at 0, where cluster 0 is picked with prob[0], etc)
"""
def pick_clusters(prob, n = 1):
    clus = np.zeros(n)
    choices = np.zeros(prob.size)
    for i in range(0, choices.size):
        choices[i] = i
    for i in range(0, n):
        clus[i] = np.random.choice(choices, p=prob)
    return clus     

"""
Centers each column using the empirical mean, and returns the centered
matrix and the emp_mean
"""
def center_mat(A):
    m = np.zeros(A.shape[0])
    for col in range(0, A.shape[1] - 1):
        m = m + A[:, col]
    if A.shape[1] != 0:
        m = m / A.shape[1]
    for col in range(0, A.shape[1] - 1):
        A[:, col] = A[:, col] - m
    return A, m

"""
Calculates and returns the empirical variance of a matrix
"""
def emp_variance(A, m):
    s = 0
    for col in range(0, A.shape[1] -1):
        Ai = A[:, col]
        Aibar = Ai - m
        Aibar_Aibar = np.multiply(Aibar, Aibar.T)
        for j in range(0, A.shape[0] - 1):
            s = s + Aibar_Aibar[j]
    var = 0
    if A.shape[1] > 0:
        var = s/(A.shape[0] - 1)
    return var
    
    
