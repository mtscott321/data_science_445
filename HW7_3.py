# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:13:13 2020

@author: mtsco
"""
import datscifuncs as dsf
import numpy as np
from matplotlib import pyplot as plt, figure as fig

k = 2
p1 = 0.3
D = 20
n = 50
sig = np.ones((D, k))
mus = np.ones((D, k))


probs = [p1, 1-p1]
A, clus = dsf.fancy_from_probs(D, n, sig, mus, probs)

plt.figure(figsize = (20, 10))
plt.plot(A[0, :], A[1, :], 'ro')
plt.show()