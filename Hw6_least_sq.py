# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:44:22 2020

@author: mtsco
"""
#%%
"""
Import needed modules
"""
import numpy as np
import math
import random as rd
from matplotlib import pyplot as plt, figure as fig
from scipy import integrate
#%%
def least_sq(mat):
    U, S, Vt = SVD(mat)
    y_arr = np.array(y)
    c = np.zeros((1,d))
    
    #in steps, calculates the sum of ((uiT*yi)/oi)*vi (this is hard to write without LaTeX)
    for i in range(0, d):
        ut_b = np.matrix.dot(U[:,i].T, y_arr)
        ut_b_over_sig = ut_b*1.0 / S[i, i]
        t = np.array(ut_b_over_sig)
        term = np.matrix.dot(t, Vt.T[:,i])
        c = c + term
    del y_arr, U, S, Vt
    return c
#%%
    #defines the error to be added to each Y value 
def err():
    return sig * rd.randrange(-1, 1)
   
#the function determining the actual values of the Y array
def f(v):
    return np.around(err() + math.sin(v), 2)

#constructing the X and Y arrays based on the function
def make_array(l = m):
    temp_a = np.zeros((l, d))
    prev = start
    for i in range (0, l):
        prev = prev + dx
        x.append(np.around(prev, 2))
        y.append(f(prev))
        for j in range(0, d):
            #A[i, j] = np.around(prev**j, 2)
            temp_a[i, j] = np.around(prev**j, 2) 
    return temp_a

#returns the x solution to Ax = b
def least_sq(mat = A):
    U, S, Vt = SVD(mat)
    y_arr = np.array(y)
    c = np.zeros((1,d))
    
    #in steps, calculates the sum of ((uiT*yi)/oi)*vi (this is hard to write without LaTeX)
    for i in range(0, d):
        ut_b = np.matrix.dot(U[:,i].T, y_arr)
        ut_b_over_sig = ut_b*1.0 / S[i, i]
        t = np.array(ut_b_over_sig)
        term = np.matrix.dot(t, Vt.T[:,i])
        c = c + term
    del y_arr, U, S, Vt
    return c

#based on the least squares result and the real y vals, returns the error
def least_sq_error_discrete(xvals, result):
    s = 0
    for i in range(0, m):
        s = s + (f(xvals[i]) - result[i])**2
    return s

def A_for_x(xval):
    a_small = np.zeros((1, d))
    for i in range(0,d):
        a_small[0, i] = xval**d
    return a_small

#returns the error in the least squares approx for continuously defined functions
def least_sq_error_cont(xvals, c, result):
    def squid(w):
        q = f(w) - np.matrix.dot(A_for_x(w), result)
        return q**2
    val_squared = integrate.quad(lambda w: squid(w), -1, 1)[0]
    return pow(val_squared, 0.5)

#multiplies three matrices together
def multiply_three(A, B, C):
    BC = np.matrix.dot(B, C)
    ABC = np.matrix.dot(A, BC)
    del BC
    return ABC


#normalizes a matrix using the Euclidean/L2 norm
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
        
#rounds all the values in an array using np.around()
def round_all_vals(mat, x=2):
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            mat[i, j] = np.around(mat[i, j], x)
    return mat

#calculates the singular value decomposition of matrix A
def SVD(A):
    #define useful values
    n = d
    min_mn = d
    if d > m:
        min_mn = m
    
    
    #get At*A
    a_t_a = np.dot(np.matrix.transpose(A), A)
    
    #get the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(a_t_a)   

    #this gets rid of any errors from calculating the eigenvalues as very small negative numbers
    vals = np.around(vals, 8)

    #computing sigma and V
    sigma = np.zeros((m, n))
    V_temp = np.zeros((n,n))
    
    #making sigma
    for j in range (0, min_mn):
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
    Ut = np.zeros((m, m))
    
    #computing U
    for j in range(0, min_mn):
        if sigma[j,j] != 0:
            Ut[j,:] = (1/sigma[j,j])*np.matrix.dot(A, V[:,j])

    U_norm = euclidean_normalized(np.matrix.transpose(Ut))
    Vt = np.matrix.transpose(V)
    
    return U_norm, sigma, Vt
