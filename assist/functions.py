"""
some functions
author: li zeng
"""

import numpy as np
import scipy

# loss function
def loss_fun(f,y):
    return np.mean(np.log(1+np.exp(-y*f)))    

# line search
def line_search(sharedK,F_train,ytrain,Kdims,pars,sele_loc=None):
    [m,beta,c] = pars    
    
    if sele_loc is None:
        sele_loc = np.array(range(len(ytrain)))
    # get K
    nrow = Kdims[0]
    width = nrow**2
    Km = sharedK[(m*width):((m+1)*width)].reshape((nrow,nrow))
    Km = Km[np.ix_(sele_loc,sele_loc)]
    
    b = Km.dot(beta)+c
    # line search function
    def temp_fun(x):
        return np.log(1+np.exp(-ytrain*(F_train+ x*b))).sum()
    out = scipy.optimize.minimize_scalar(temp_fun)
    if not out.success:
        print("warning: minimization failure")
    return out.x

# sampling 
def subsamp(y,col,fold=3):
    grouped = y.groupby(col)
    out = [list() for i in range(fold)]
    for i in grouped.groups.keys():
        ind = grouped.get_group(i)
        n = ind.shape[0]
        r = list(range(0,n+1,n//fold))
        r[-1] = n+1
        # permute index
        perm_index = np.random.permutation(ind.index)
        for j in range(fold):
            out[j] += list(perm_index[r[j]:r[j+1]])
    return out