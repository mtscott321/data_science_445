# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:13:13 2020

@author: mtsco
"""
import datscifuncs as dsf
import numpy as np
from matplotlib import pyplot as plt, figure as fig

#%%
k = 2
p1 = 0.3
D = 20
n = 50
#determines the scale of the whole graph
var = 1
sig = np.ones((D, k)) * var
mus = np.ones((D, k)) * 5

def cluster_mus(mus, k, a):
    D = mus.shape[0]
    for i in range(0, k):
        #choosing the center randomly from a D-dim normal with mu = 0 and std = the random int
        mus[:, i] = np.random.rand(D) * np.random.randint(k+1)
    return mus

#mus = cluster_mus(mus, k, 20)
mus[:, 1] = np.zeros(D)


probs = [p1, 1-p1]
A, clus = dsf.fancy_from_probs(D, n, sig, mus, probs)

def k_means(data, corr_clus, k):
    clus = dsf.choose_first_clus(data, k)
    clus_centers = np.zeros((D, k))
    for i in range(0, 100):
        clus_centers = dsf.calculate_means(clus, data, k)
        clus = dsf.choose_clusters(clus_centers, data)
    p_err = dsf.k_means_err_perc(clus, corr_clus, data.shape[1])
    err = dsf.k_means_err(data, clus, clus_centers)
    return p_err, err

p_err, err = k_means(A, clus, k)
    



plt.figure(figsize = (20, 10))
plt.plot(A[0, :], A[1, :], 'ro')
plt.show()
