# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:33:02 2020

@author: mtsco
"""

import datscifuncs as dsf
from datscifuncs import normal_stnd_dist_mat
import numpy as np


#%%

def PCA(A, k):
    U, sig, Vt = dsf.SVD(A)
    for i in range(k + 1, A.shape[0] - 1):
        U[i, :] = np.zeros(A.shape[0])
    projA = np.dot(U.T, A)
    return projA
       

#%%
D = 200
n = 400
k = 2
A = normal_stnd_dist_mat(D, n)
Abar, emp_mean = dsf.center_mat(A)
v = dsf.emp_variance(A, emp_mean)
U, sig, Vt = dsf.SVD(A)
projA = PCA(Abar, k)
projAbar, proj_emp_mean = dsf.center_mat(projA)
proj_v = dsf.emp_variance(projA, emp_mean)
print(proj_v)
sig_k_sq = sig[k,k] *sig[k,k]
print(sig[k,k])
print(sig_k_sq)

#%%
D = 20
n = 40
k = 4
A = normal_stnd_dist_mat(D, n)
U, sig, Vt = dsf.SVD(A)
#center A
Abar, emp_mean = dsf.center_mat(A)
projA = PCA(Abar, k)

newAbar,newemp_mean = dsf.center_mat(projA)
v = dsf.emp_variance(Abar, newemp_mean)

print(v)
print(sig[k, k] * sig[k, k])
#%%
print(A[:, 0]) #this prints the first column

#%%
A = normal_stnd_dist_mat(3,2)
print(A)
U, sig, Vt = dsf.SVD(A)
print(dsf.multiply_three(U, sig, Vt))

#%%




