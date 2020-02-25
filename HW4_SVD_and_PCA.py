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
from matplotlib import pyplot as plt

#%%
"""
Define the array of values for {(xi, yi)}1:m
"""

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


x = []
y = []
m = 30
d = 4
r = 30
n = d

A = np.zeros((m, d))


max_mn = m
min_mn = n
if n > m:
    max_mn = n
    min_mn = m
    
start = -1
end = 1
sig = 0.05
dx = (end - start)*1.0 /m
    
def err():
    return sig * rd.randrange(-1, 1)
    
def f(val):
    return np.around(err() + math.sin(val), 2)

#main code for this section
def make_array():
    prev = start
    x = []
    for i in range (0, m):
        prev = prev + dx
        x.append(np.around(prev, 2))
        print(x)
        y.append(f(prev))
        for j in range(0, d):
            A[i, j] = np.around(prev**j, 2)

#%%
"""
Plotting x and y values
"""
plt.xlabel("X values")
plt.ylabel("sin(x) in radians")
plt.plot(x, y)
plt.show
#%%

def least_sq():
    U, S, Vt = SVD(A)
    y_arr = np.array(y)
    c = np.zeros((1,d))
    for i in range(0, d):
        ut_b = np.matrix.dot(U[:,i].T, y_arr)
        ut_b_over_sig = ut_b*1.0 / S[i, i]
        t = np.array(ut_b_over_sig)
        term = np.matrix.dot(t, Vt.T[:,i])
        c = c + term
    return c
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
for i in range(0, 10):
     make_array()
     result = np.dot(A, least_sq().T)       
     all_errors.append(least_sq_error(x, result)) 

#%%
def least_sq_error(xvals, result):
    s = 0
    for i in range(0, m):
        s = s + (f(xvals[i]) - result[i])**2
    return s
#%%
"""
Define A
"""

A = np.matrix([[3, 5, 6, 7],
              [5, 9, 0, 1],
              [6, 6, 6, 4]])
print(A)

m = A.shape[0]
n = A.shape[1]
print(m)
print(n)

max_mn = m
min_mn = n
if n > m:
    max_mn = n
    min_mn = m


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
        

def round_all_vals(mat, x=2):
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            mat[i, j] = np.around(mat[i, j], x)
    return mat
    
def SVD(A):
    #get At*A
    a_t_a = np.dot(np.matrix.transpose(A), A)
    
    #get the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(a_t_a)   

    #this gets rid of any errors from calculating the eigenvalues as very small negative numbers
    vals = np.around(vals, 8)
    """
    print("these are the eigenvecs")
    print(round_all_vals(vecs))
    """
    #computing sigma and V
    sigma = np.zeros((m, n))
    V_temp = np.zeros((n,n))
    
    for j in range (0, min_mn):
        #vals is returned in the reverse order
        #this assigns the diagonal of sigma as the sqrt of the eigenvals
        sigma[j, j] = math.sqrt(vals[vals.shape[0] - 1 -j])
        
    for i in range(0, n):
        #vecs is returned in the reverse order
        #this assigns the columns of V as the eigenvectors
        temporary = vecs[i]
        """
        print("\n\nchecking stuff")
        print(temporary)
        print(temporary.shape)
        print(V_temp[i,:].shape)
        print(V_temp[i,:])
        """
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

#%%

  #  print((1/B[j,j])*np.matrix.dot(test, Ct[:,j]))
#%%
"""
A = np.matrix([[1,2,3],
              [4,5,6]])
print(A[0])
print(type(np.array(A[0])))
print(A[0])
print(np.array(A[0])[::-1])


print(np.fliplr(A))
"""