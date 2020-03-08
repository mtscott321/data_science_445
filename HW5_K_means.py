# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 01:52:59 2020

@author: mtsco
"""

# -*- coding: utf-8 -*-
"""
Madeline Scott
Mathematical and Computiational Foundations of Data Science
Homework 4, due 26 Feb 2020

This program computes the least square of a matrix, as well as the least
squares computation for sin(x) using a Vandermonde matrix.
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
from cycler import cycler

#%%
#now find the ideal way to split the selected values to minimize distance
def clusters_from_ms(means):
    new_clusters = {}
    error = 0
    for x in selected:
        mindist = [-1, -1]
        for i in range(0, len(means)):
            dist = np.linalg.norm(x[0][0]-means[i])
            if mindist[0] < 0 or mindist[0] > dist:
                #keeping track of the smallest distance and which cluster that was a part of
                mindist[0] = dist
                mindist[1] = i
        error = error  + mindist[0]
        if mindist[1] not in new_clusters:
            new_clusters[mindist[1]] = []
        new_clusters[mindist[1]].append(x[0][0])
    return new_clusters, error
            

def ms_from_clusters(clus):
    means = []
    #for each label
    for i in clus:
        #for each x corresponding to each label
        s = 0
        num = 0
        for j in clus[i]:
            s = s + j
            num = num + 1
        #don't have to worry about divide by zero bc each has at least 1
        avg = s / num
        means.append(avg)
    return means



def graph_the_points(clus, means):
    plt.figure(figsize=(20, 10))
    #this will iterate so each cluser gets a diff color up until four clusters lol
    plt.rc(cycler('color', ['co', 'mo', 'yo', 'ko']))
    #plot the x and y values
    for i in clus:
        plt.plot(np.stack(clus[i])[:,0], np.stack(clus[i])[:,1], marker = 'o')
    #now plot the means
    for m in means:
        plt.plot(m[0], m[1], 'g^')
    
    plt.show

def randomly_choose(selected):
    means = np.random.choice(np.asarrar(selected), k, replace = False)
    return means

#%%
"""
Generate the gaussian distributions
"""
D = 14
m = 5
n = 500
k = 2

temp = np.zeros(D)
temp[0] = m

g1 = np.random.multivariate_normal(np.zeros(D), np.identity(D), size=100)
g2 = np.random.multivariate_normal(temp, np.identity(D), size=100)

del temp

p1 = 0.5
p2 = 1 - p1

#selected is the data set of the n many selected values from both of the arrays
#s is a python list of lists, each of which contains [0]: np.array(D) and [1]: index in the form of an integer 0 to k
s = []

#certain there is a better way to do this but I don't know what it is
#selecting the values from the Gaussians at the probabilities given
for i in range(0, n):
    l = np.random.choice([0, 1], p = [p1, p2])
    if l == 0:
        #should the selection have replacement or not? 
        #I did with replacement, but I'm not sure which is correct
        index = np.random.choice(g1.shape[0], 1)
        s.append([g1[index], 0])
    else:
        index = np.random.choice(g2.shape[0], 1)
        s.append([g2[index], 1])

#%%
        """This is the big leap from old to new """
        
#now we're going to start doing the k-means
#pick the first middle points randomly from the set
        
c, erroror = clusters_from_ms(s)
iterations = 0

#keeping track of iterations to prevent it from just not settling ever
while erroror > 0.5 and iterations < 25:
    ms = ms_from_clusters(c)
    c, erroror = clusters_from_ms(ms)
    iterations = iterations + 1
    print(erroror)

#now we're going to assign the original clusters to the new clusters (organized by indicies in their respective arrays)
old_2_new = {}
for i in range(0, 7):
    #the key represents the index in the 



#%%
"""
Now that we have clusters, we need to find out which one corresponds to which of the actual values
"""
tracking = {}
if np.linalg.norm(ms[0]) < np.linalg.norm(ms[1]):
    tracking[0] = 1
    tracking[1] = 0
else: 
    tracking[0] = 0
    tracking[1] = 1
    
    
    
#%%
    

#%%
#for each x value in each bin of c
        #check to see if that assignment is the same as the assignment in selected
        
corr = 0
wrong = 0
for value in selected:
    
    x = value[0][0]
    bin_ = value[1]
    same = True
    for i in range(0, len(c[tracking[bin_]][0])):
        if c[tracking[bin_]][0][i] != x[i]:
            same = False
    if same:
        wrong = wrong + 1
    else:
        corr = corr + 1
#%%
print(wrong)
print(corr)
        
#%%

total_error = wrong*1.0 / n

#%%
"""
now we're gonna do the whole thing but with a lot of different n values
"""
ns = []
tot_errs = []
for n in range(4, 100):
    D = 8
    m = 5
    
    k = 2
    
    temp = np.zeros(D)
    temp[0] = m
    
    g1 = np.random.multivariate_normal(np.zeros(D), np.identity(D), size=100)
    g2 = np.random.multivariate_normal(temp, np.identity(D), size=100)
    
    del temp
    
    p1 = 0.5
    p2 = 1 - p1
    
    #selected is the data set of the n many selected values from both of the arrays
    selected = []
    
    #certain there is a better way to do this but I don't know what it is
    #selecting the values from the Gaussians at the probabilities given
    for i in range(0, n):
        l = np.random.choice([0, 1], p = [p1, p2])
        if l == 0:
            #should the selection have replacement or not? 
            #I did with replacement, but I'm not sure which is correct
            index = np.random.choice(g1.shape[0], 1)
            selected.append([g1[index], 0])
        else:
            index = np.random.choice(g2.shape[0], 1)
            selected.append([g2[index], 1])
            
    
    all_means = []
    js = np.random.choice(len(selected), k, replace = False)
    for i in range (0, k):
        #all the indices are because the arrays are stored as 2d 1xn arrays rather than 1D
        all_means.append(selected[js[i]][0][0])
    c, error = clusters_from_ms(all_means)
    iterations = 0
    
    #keeping track of iterations to prevent it from just not settling ever
    while error > 0.5 and iterations < 1000:
        ms = ms_from_clusters(c)
        c, error = clusters_from_ms(ms)
        iterations = iterations + 1
    tracking = {}
    if np.linalg.norm(ms[0]) < np.linalg.norm(ms[1]):
        tracking[0] = 1
        tracking[1] = 0
    else: 
        tracking[0] = 0
        tracking[1] = 1
    
    corr = 0
    wrong = 0
    for value in selected:
        
        x = value[0][0]
        bin_ = value[1]
        same = True
        for i in range(0, len(c[tracking[bin_]][0])):
            if c[tracking[bin_]][0][i] != x[i]:
                same = False
        if same:
            wrong = wrong + 1
        else:
            corr = corr + 1
    
    total_error = wrong*1.0 / n
    
    ns.append(n)
    tot_errs.append(total_error)

#%%
plt.plot(ns, tot_errs)
plt.xlabel("N values")
plt.ylabel("Percent error")
    
    
#%%
all_means = []
js = np.random.choice(len(selected), k, replace = False)
for i in range (0, k):
    #all the indices are because the arrays are stored as 2d 1xn arrays rather than 1D
    all_means.append(selected[js[i]][0][0])
#%%
c, erroror = clusters_from_ms(all_means)
ms = ms_from_clusters(c)
print(erroror)
#%%


c, erroror = clusters_from_ms(ms)
ms = ms_from_clusters(c)
print(erroror)

    
#%%

print(c)
print("n\n\n\n\n\n")
print(all_means)
graph_the_points(c, all_means)



tempms = ms_from_clusters(c)

c1, errorororor = clusters_from_ms(tempms)

graph_the_points(c1, tempms)

print(c)
print("\n\n\n\n\n")
print(c1)



        
plt.figure(figsize=(20, 10))
xvals = g1[:,0]
print(xvals)
plt.plot(xvals, g1[:,1], 'ro')
plt.plot(g2[:,0], g2[:,1], 'go')
plt.show

