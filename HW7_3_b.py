# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:25:15 2020

@author: mtsco
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:13:13 2020

@author: mtsco
"""
import datscifuncs as dsf
import numpy as np
from matplotlib import pyplot as plt, figure as fig



def m_and_err(m, probs, D, n, sig, k):
    mus = np.array([[0, m], [0, 0]])
    A, clus = dsf.fancy_from_probs(D, n, sig, mus, probs)
    A = mult_w(A, clus, probs)
    p_err, trash = k_means(A, clus, k)
    return p_err

def mult_w(data, corr_clus, probs):
    n = data.shape[1]
    for i in range(0, n):
        j = int(corr_clus[i])
        data[:, i] = data[:, i] * probs[j]
    return data

def k_means(data, corr_clus, k):
    clus = dsf.choose_first_clus(data, k)
    clus_centers = np.zeros((D, k))
    for i in range(0, 100):
        clus_centers = dsf.calculate_means(clus, data, k)
        clus = dsf.choose_clusters(clus_centers, data)
    p_err = dsf.k_means_err_perc(clus, corr_clus, data.shape[1])
    err = dsf.k_means_err(data, clus, clus_centers)
    return p_err, err

#%%
k =  2
p1 = 0.3
D = 2
n = 50
probs = [p1, 1-p1]
#each column of sig is a stdev
sig = np.array([[1, 1], [2, 2]])
ms = []
errs = []
for m in np.linspace(0, 1, 50):
    ms.append(m)
    e = 0
    for j in range(0, 1):
        e = e + m_and_err(m, probs, D, n, sig, k)
    errs.append(e / 1)
    
plt.figure(figsize = (20, 10))
plt.plot(ms, errs, 'ro')
plt.show()
