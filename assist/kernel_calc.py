"""
file: program to calculate kernel matrices
author: lizeng
"""

from sklearn import metrics
import numpy as np
import sys

def get_kernels(X,Y,inputs):
    M = inputs.Ngroup
    N1 = X.shape[0]
    N2 = Y.shape[0]
    out = np.zeros([N1,N2,M])
    
    ct = 0
    for value in inputs.pred_sets.values:
        genes = value.split(" ")
        a = X[genes]
        b = Y[genes]
        if inputs.kernel=='rbf':
            out[:,:,ct] = metrics.pairwise.rbf_kernel(X= a,Y =b,gamma= 1/len(genes))
        elif inputs.kernel[:4]=='poly':
            deg = int(inputs.kernel[4])
            out[:,:,ct] = metrics.pairwise.polynomial_kernel(X = a, Y= b, degree = deg, gamma = 1/len(genes))
        else:
            print("wrong kernel option: "+inputs.kernel+'\n')
            sys.exit(-3)
        ct += 1
    return out