# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:31:42 2020

@author: mtsco
"""

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
import random as rd
from matplotlib import pyplot as plt, figure as fig


#%%
"""
Define globals
"""

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

#globals, X, Y arrays and the matrix
x = []
y = []
m = 30
d = 4
A = np.zeros((m, d))
    
#used to compute the X, Y arrays
start = -1
end = 1
sig = 0.1
dx = (end - start)*1.0 /m

#%%

#defines the error to be added to each Y value 
def err():
    return sig * rd.randrange(-1, 1)
   
#the function determining the actual values of the Y array
def f(val):
    return np.around(err() + math.sin(val), 2)

#constructing the X and Y arrays based on the function
def make_array():
    prev = start
    for i in range (0, m):
        prev = prev + dx
        x.append(np.around(prev, 2))
        print(x)
        y.append(f(prev))
        for j in range(0, d):
            A[i, j] = np.around(prev**j, 2)

#returns the x solution to Ax = b
def least_sq():
    U, S, Vt = SVD(A)
    y_arr = np.array(y)
    c = np.zeros((1,d))
    
    #in steps, calculates the sum of ((uiT*yi)/oi)*vi (this is hard to write without LaTeX)
    for i in range(0, d):
        ut_b = np.matrix.dot(U[:,i].T, y_arr)
        ut_b_over_sig = ut_b*1.0 / S[i, i]
        t = np.array(ut_b_over_sig)
        term = np.matrix.dot(t, Vt.T[:,i])
        c = c + term
    return c

#based on the least squares result and the real y vals, returns the error
def least_sq_error(xvals, result):
    s = 0
    for i in range(0, m):
        s = s + (f(xvals[i]) - result[i])**2
    return s

#multiplies three matrices together
def multiply_three(A, B, C):
    BC = np.matrix.dot(B, C)
    ABC = np.matrix.dot(A, BC)
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

#%%
"""
Plotting x and y values
"""
plt.xlabel("X values")
plt.ylabel("sin(x) in radians")
plt.plot(x, y)
plt.show

#%%
print(c)

result = np.dot(A, c.T)

plt.plot(x, result)
plt.show

#%%
"""
Running the code several times for multiple m values and plotting the mean and std_dev
"""
all_errors = []
for i in range(0, 100):
    #have to make x and y empty again
    x = []
    y = []
    make_array()
    result = np.dot(A, least_sq().T)       
    all_errors.append(least_sq_error(x, result)) 

mean = np.average(all_errors)
std = np.std(all_errors)
print("The mean value of the error for m = %d is: %.3f and the STD = %0.3f" %(m, mean, std))


#%%

def test1(A):
    print(A)
    
    Q, B, C = SVD(A)
    print("\nThis is U")
    print(Q)
    print("\nThis is Sigma")
    print(B)
    print("\nThis is V")
    print(C.T)
    print("\nThis is their product")
    print(round_all_vals(multiply_three(Q, B, C), 0))
