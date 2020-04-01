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
part 1d
"""
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
part 1c
"""

k = 3
A = dsf.normal_stnd_dist_mat(50, 100)
U, sig, Vt = dsf.SVD(A)
projA = dsf.PCA(A, k)
var = dsf.emp_variance(A, k)
sig_sq = sig[k-1,k-1]*sig[k-1, k-1]

print(A.shape[1] * var)

print(sig_sq)
