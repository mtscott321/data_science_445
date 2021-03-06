# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:33:54 2020

@author: mtsco
"""
import math
import numpy as np
import statistics
import scipy

def round_all_vals(mat, x=2):
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            mat[i, j] = np.around(mat[i, j], x)
    return mat

def multiply_three(A, B, C):
    BC = np.matrix.dot(B, C)
    ABC = np.matrix.dot(A, BC)
    del BC
    return ABC

def euclidean_normalize_cols(A):
    for j in range(0, A.shape[1]):
        s = 0
        for i in range(0, A.shape[0]):
            s = s + float(A[i, j])*float(A[i, j])
        norm = math.sqrt(np.around(s, 8))
        for i in range(0, A.shape[0]):
            if norm != 0:
                A[i,j] = A[i,j]/norm
    return A

def SVD(A):
    [D, n] = A.shape
    min_Dn = D
    if D > n:
        min_Dn = n
    #get At*A
    a_t_a = np.dot(np.matrix.transpose(A), A)
    
    #get the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(a_t_a)   

    #this gets rid of any errors from calculating the eigenvalues as very small negative numbers
    vals = np.around(vals, 8)

    #computing sigma and V
    sigma = np.zeros((D, n))
    V_temp = np.zeros((n,n))
    
    #making sigma
    for j in range (0, min_Dn):
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
    Ut = np.zeros((D, D))
    
    #computing U
    for j in range(0, min_Dn):
        if sigma[j,j] != 0:
            Ut[j,:] = (1/sigma[j,j])*np.matrix.dot(A, V[:,j])

    U_norm = euclidean_normalize_cols(np.matrix.transpose(Ut))
    Vt = np.matrix.transpose(V)
    
    return U_norm, sigma, Vt

"""
returns a matrix of values, each ith column (out of n) chosen from a normal 
distribution with mean mus[i] and standard deviation sig[i]
"""

def normal_dist_mat(D, n, sig, mus):
    A = np.zeros((D, n))
    for col in range(0, n):
        new_col = np.random.normal(loc = mus[col], scale = sig[col], size = (D))
        A[:, col] = new_col
    return A

def normal_stnd_dist_mat(D, n):
    return normal_dist_mat(D, n, sig = np.ones([n]), mus = np.zeros([n]))

"""
clus will be a 1xn np array, where each clus[i] is the label of the cluster that the ith data point
will belong to. Example: clus[4] = 2, so the 4th data point A[:, 4] will be drawn from cluster #4.
sigma is a Dxk array (where k is the number of clusters), where each sigma[i, j] is the stddev in the ith dimension for the jth cluster
mus is a Dxk (where k is the number of clusters) array where each column is the average value of that cluster
"""
def fancy_normal(D, n, sig, mus, clus):
    A = np.zeros((D, n))
    #picking n points
    for i in range(0, n):
        j = int(clus[i])
        #randomly choosing a D-dim vector from a D-dim normal dist
        #then scaling by the stddev of each dim after shifting to the cluster average
        centered = np.random.randn(D) - mus[:, j]
        for k in range(0, D):
            centered[k] = sig[k, j] * centered[k]
        
        A[:, i] = centered
    return A

def fancy_from_probs(D, n, sig, mus, probs):
    clus = pick_clusters(probs, n)
    return fancy_normal(D, n, sig, mus, clus), clus

"""
n many times, picks a cluster from the prob.size posibilities, with each ith 
cluster having a probability of prob[i] of being picked. Returns an array of the 
selected clusters (starting at 0, where cluster 0 is picked with prob[0], etc)
"""
def pick_clusters(prob, n = 1):
    clus = np.zeros(n)
    choices = np.zeros(len(prob))
    for i in range(0, choices.size):
        choices[i] = i
    for i in range(0, n):
        clus[i] = np.random.choice(choices, p=prob)
    return clus     

"""
Centers each column using the empirical mean, and returns the centered
matrix and the emp_mean
"""
def center_mat(A):
    m = np.zeros(A.shape[0])
    for col in range(0, A.shape[1]):
        m = m + A[:, col]
    if A.shape[1] != 0:
        m = m / A.shape[1]
    for col in range(0, A.shape[1]):
        A[:, col] = A[:, col] - m
    return A, m

"""
Calculates and returns the empirical variance of a matrix. Inputs should be a matrix, 
and the k value for the PCA
"""
def emp_variance(A, k, center = True):
    cA, trash = center_mat(A)
    U, sig, Vt = SVD(cA)
    Uk = U[:, k-1]
    Z = []
    for col in range(0, cA.shape[1]):
        cAi = cA[:, col]
        dot = np.dot(cAi, Uk)
        Z.append(dot)
    m = statistics.mean(Z)
    s = 0
    for i in range(0, cA.shape[1]):
        temp = Z[i] - m
        s = s + temp*temp
    var = -1
    #this should always happen, but need to avoid divide by zero
    if A.shape[1] > 1:
        var = s/(A.shape[1] - 1)
    return var

"""
Projects matrix A into a k-dimensional subspace with principal component
basis. Returns the projected matrix.
"""
def PCA(A, k, center = True):
    cA, trash = center_mat(A)
    U, sig, Vt = SVD(cA)
    for i in range(k, cA.shape[0]):
        U[:, i] = 0
    projA = np.dot(U.T, cA)
    return projA, U

def PCA_error(A, k):
    cA, trash = center_mat(A)
    projA, U = PCA(A, k)
    s = 0
    for col in range(0, cA.shape[1]):
        temp = cA[:, col] - np.dot(np.dot(U, U.T), cA[:, col])
        s = s + (np.linalg.norm(temp))**2
    err = 0
    if cA.shape[1] > 0:
        err = s / cA.shape[1]
    return err
        
def L2_norm(f, a, b):
    integral = L2_dp(f, f, a, b)
    return math.sqrt(integral)

def L2_dp(f, g, a, b):
    integral, err = scipy.integrate.quad(lambda x: f(x)*g(x), a, b)
    return integral

"""
K-means functions below
"""
    
    #now find the ideal way to split the selected values to minimize distance
def clusters_from_ms(means, selected):
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
            
"""
clus_centers is a Dxk matrix where k is the number of clusters. Each jth column is the mean 
value for the jth cluster
data is a Dxn matrix where each ith column is the ith data point. 
new clusters is a 1xn matrix where clus[i] = the cluster to which the ith data point has been assigned.
The value will indicate the column of clus_centers.
"""
def choose_clusters(clus_centers, data):
    n = data.shape[1]
    k = clus_centers.shape[1]
    new_clus = np.zeros(n)
    for i in range(0, n):
        curr_clus = -1
        min_dist = -1
        for j in range(0, k):
            curr_dist = np.linalg.norm(data[:, i] - clus_centers[:, j])
            if curr_dist < min_dist or min_dist < 0:
                min_dist = curr_dist
                curr_clus = j
        new_clus[i] = curr_clus
    return new_clus

"""
clusters is a 1xn array where the ith value is the cluster to which the ith data 
point belongs.
data is a Dxn matrix where each ith column is the ith data point. 
"""
def calculate_means(clusters, data, k):
    D = data.shape[0]
    n = data.shape[1]
    clus_centers = np.zeros((D, k))
    #for each cluster
    for i in range(0, k):
        cluster_sum = np.zeros(D)
        sum_num = 0
        #check to see if the ith data is in the cluster, and if it is add to sum
        for j in range(0, n):
            if clusters[j] == i:
                cluster_sum = cluster_sum + data[:, j]
                sum_num  = sum_num + 1
        clus_centers[:, i] = cluster_sum / sum_num
    return clus_centers

def k_means_err_perc(selected_clus, actual_clus, n):
    incorr = 0
    for i in range(0, n):
        if selected_clus[i] != actual_clus[i]:
            incorr = incorr + 1
    perc_err  = 1.0*incorr/n
    return perc_err
        

def k_means_err(data, selected_clus, clus_centers):
    n = data.shape[1]
    k = clus_centers.shape[1]
    err_sum = 0
    for i in range(0, k):
        for j in range(0, n):
            if selected_clus[j] == i:
                dist = np.linalg.norm(data[:, j] - clus_centers[:, i])
                err_sum = err_sum + dist**2
    return err_sum

def choose_first_clus(data, k):
    n = data.shape[1]
    clusters = np.zeros(n)
    for i in range(0, n):
        clusters[i] = np.random.randint(k+1)
    return clusters
        
    
    
    
    
