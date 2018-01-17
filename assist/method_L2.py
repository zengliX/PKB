"""
L2 penalty method
author: li zeng
"""

import numpy as np
from assist.deriv_calcu import calcu_h, calcu_q
import multiprocessing as mp

# function to parallelized

def paral_fun_L2(sharedK,m,nrow,h,q,Lambda,sele_loc):
    # get K
    width = nrow**2
    Km = sharedK[(m*width):((m+1)*width)].reshape((nrow,nrow))
    Km = Km[np.ix_(sele_loc,sele_loc)]
    # working Lambda
    new_Lambda= len(sele_loc)*Lambda

    # calculte values
    K2 = np.append(Km,np.ones([len(sele_loc),1]),axis=1)
    L_mat = np.diag(list(np.repeat(new_Lambda,len(sele_loc)))+[0])
    eta = -np.linalg.solve((K2.T).dot(np.diag(q/2)).dot(K2)+L_mat,(K2.T).dot(h/2))
    beta = eta[:-1]
    c = eta[-1]
    temp = K2.dot(eta)
    #val = temp.dot(np.diag(q/2)).dot(temp)+temp.dot(h)
    val = temp.dot(np.diag(q/2)).dot(temp)+temp.dot(h) + new_Lambda*np.sum(beta**2)
    return [val,[m,beta,c]]


# find Lambda
def find_Lambda_L2(K_train,F_train,ytrain,Kdims,C=2):
    h = calcu_h(F_train,ytrain)
    q = calcu_q(F_train,ytrain)
    # max(|Km*eta|)/N for each Km
    eta = h/q
    w_half = np.diag(np.sqrt(q/2))
    eta_tilde = w_half.dot(eta - eta*q/q.sum())
    
    l_list = [] # list of lambdas from each group
    for m in range(Kdims[1]):
        Km = K_train[:,:,m]
        Km_tilde = w_half.dot(Km - np.ones([Kdims[0],1]).dot((q/q.sum()).dot(Km).reshape([1,Kdims[0]])))
        d = np.linalg.svd(Km_tilde)[1][0]
        l = d*(np.sqrt(np.sum(eta_tilde**2)) - C)/(C*Kdims[0])
        if l > 0:
            l_list.append(l)
        else:
            l_list.append(0.01)
    return np.percentile(l_list,20)


# one boosting iteration for second order method
def oneiter_L2(sharedK,F_train,ytrain,Kdims,Lambda,ncpu = 1,\
               parallel=False,sele_loc = None,group_subset = False):
    # whether stochastic gradient boosting
    if sele_loc is None:
        sele_loc = np.array(range(len(ytrain)))
        
    # calculate derivatives h,q
    h = calcu_h(F_train,ytrain)
    q = calcu_q(F_train,ytrain)
    
    # identify best fit K_m
    if not parallel: ncpu =1
        # random subset of groups
    mlist = range(Kdims[1])
    if group_subset:
        mlist= np.random.choice(mlist,min([Kdims[1]//3,100]),replace=False)
        
    pool = mp.Pool(processes =ncpu,maxtasksperchild=300)
    results = [pool.apply_async(paral_fun_L2,args=(sharedK,m,Kdims[0],h,q,Lambda,sele_loc)) for m in mlist]
    out = [res.get() for res in results]
    pool.close()
    return out[np.argmin([x[0] for x in out])][1]
