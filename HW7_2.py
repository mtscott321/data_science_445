# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:33:15 2020

@author: mtsco
"""

import datscifuncs as dsf
import math

"""
Computes a Gram-Schmidt projection of the function f onto the subspace defined
by the basis of functions in basis, defined over the L2 space with integrals from a to b
"""
def gram_schmidt(basis, a, b):
    new = []
    for i in range(0, len(basis)):
        temp = basis[i]
        for w in new:
            def t(x):
                return temp(x) - dsf.L2_dp(w, basis[i], a, b) * w(x)
            temp = t
        temp_norm = dsf.L2_norm(temp, a, b)
        def nt(x):
            return temp(x) / temp_norm
        norm_temp = nt
        new.append(norm_temp)
    return new

def check(funs):
    ints = []
    for f in funs:
        ints.append(dsf.L2_norm(f, 0, 1))
    return ints

def one(x):
    return 1
def x(x):
    return x
def x_sq(x):
    return x*x



funs = [one, x, x_sq]
print(check(funs))

gram_schmidt(funs, 0, math.pi)
    
    
    
    
#%%
