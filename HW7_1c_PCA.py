# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:33:02 2020

@author: mtsco
"""

import datscifuncs as dsf
from datscifuncs import normal_stnd_dist_mat
import numpy as np
from matplotlib import pyplot as plt, figure as fig

#%%
"""

#%%
D = 30
n = 400
k = 2
A = normal_stnd_dist_mat(D, n)
Abar, emp_mean = dsf.center_mat(A)
v = dsf.emp_variance(Abar, emp_mean)
U, sig, Vt = dsf.SVD(A)
projA = dsf.PCA(Abar, k)
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
projA = dsf.PCA(Abar, k)

newAbar,newemp_mean = dsf.center_mat(projA)
v = dsf.emp_variance(newAbar, emp_mean)


print(v)
print(sig[k-1, k-1])
print(sig[k-1, k-1] * sig[k-1, k-1])
#%%
print(A[:, 0]) #this prints the first column

#%%
A = dsf.normal_stnd_dist_mat(3,2)
print(A)
U, sig, Vt = dsf.SVD(A)
print(dsf.multiply_three(U, sig, Vt))
"""

#%%

def error_with_changing_k(D, n, kstart, kend):
    A = dsf.normal_stnd_dist_mat(D, n)
    errors = []
    ks = []
    for k in range(kstart, kend):
        ks.append(k)
        errors.append(dsf.PCA_error(A, k))
    return ks, errors


ks, errors = error_with_changing_k(50, 300, 1, 50)
plt.figure(figsize = (20, 10))
plt.plot(ks, errors, 'ro')
plt.xlabel("K value")
plt.ylabel("PCA error")
plt.show()
        

#%%

"""

k = 3
A = dsf.normal_stnd_dist_mat(50, 100)
U, sig, Vt = dsf.SVD(A)
projA = dsf.PCA(A, k)
var = dsf.emp_variance(A, k)
sig_sq = sig[k-1,k-1]*sig[k-1, k-1]

print(A.shape[1] * var)
print(sig_sq)

print(dsf.PCA_error(A, k))
"""