"""
CV to determine optimal number of iterations
author: li zeng
"""

from assist.functions import loss_fun, line_search, subsamp
import numpy as np
import pandas as pd
#import time
from assist.method_L1 import oneiter_L1
from assist.method_L2 import oneiter_L2
from matplotlib import pyplot as plt

def CV_PKB(inputs,sharedK,K_train,Kdims,Lambda,nfold=3,ESTOP=30,ncpu=1,parallel=False,gr_sub=False,plot=False):
    ########## split data ###############
    test_inds = subsamp(inputs.train_response,inputs.train_response.columns[0],nfold)
    temp = pd.Series(range(inputs.Ntrain),index= inputs.train_response.index)
    folds = []
    for i in range(nfold):
        folds.append([ temp[test_inds[i]].values, np.setdiff1d(temp.values,temp[test_inds[i]].values)])
    
    ########## initiate for each fold ###############
    ytrain_ls = [np.squeeze(inputs.train_response.iloc[folds[i][1]].values) for i in range(nfold)]
    ytest_ls = [np.squeeze(inputs.train_response.iloc[folds[i][0]].values) for i in range(nfold)]
    
    Ftrain_ls = []
    Ftest_ls = []
    test_err_ls = [[] for i in range(nfold)]
    test_loss_ls = [[] for i in range(nfold)]
    for i in range(nfold):
        F0 = np.log((ytrain_ls[i] == 1).sum()/(ytrain_ls[i] ==-1).sum())      
        if F0==0: F0+=10**(-2)
        Ftrain_ls.append(np.repeat(F0,len(folds[i][1])))  # keep track of F_t(x_i) on training data
        Ftest_ls.append(np.repeat(F0,len(folds[i][0]))) #keep track of F_t(x_i) on testing data
        #train_loss.append(loss_fun(F_train,ytrain))
        test_err_ls[i].append((np.sign(Ftest_ls[i]) != ytest_ls[i]).sum()/len(ytest_ls[i]))
        test_loss_ls[i].append(loss_fun(Ftest_ls[i],ytest_ls[i]))    
    
    opt_iter = 0
    min_loss = prev_loss =  np.mean([x[0] for x in test_loss_ls])
    ave_err = [np.mean([x[0] for x in test_err_ls])]
    ave_loss = [prev_loss]
    ########## boosting for each fold ###############
    # time0 = time.time()
    print("-------------------- CV -----------------------")
    print("iteration\tMean test err\tMean test loss")
    for t in range(1,inputs.maxiter+1):
        # one iteration
        for k in range(nfold):
            if inputs.method == 'L2':
                [m,beta,c] = oneiter_L2(sharedK,Ftrain_ls[k],ytrain_ls[k],Kdims,Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                sele_loc=folds[k][1])    
            if inputs.method == 'L1':
                [m,beta,c] = oneiter_L1(sharedK,Ftrain_ls[k],ytrain_ls[k],Kdims,Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                sele_loc=folds[k][1],group_subset = gr_sub) 
    
            # line search
            x = line_search(sharedK,Ftrain_ls[k],ytrain_ls[k],Kdims,[m,beta,c],sele_loc=folds[k][1])
            beta *= x
            c *= x
    
            # update lists
            Ftrain_ls[k] += (K_train[:,:,m][np.ix_(folds[k][1],folds[k][1])].dot(beta) + c)*inputs.nu
            Ftest_ls[k] += (K_train[:,:,m][np.ix_(folds[k][0],folds[k][1])].dot(beta)+ c)*inputs.nu
            test_err_ls[k].append((np.sign(Ftest_ls[k])!=ytest_ls[k]).sum()/len(ytest_ls[k]))
            new_loss = loss_fun(Ftest_ls[k],ytest_ls[k])
            test_loss_ls[k].append(new_loss)
        
        # save iteration
        cur_err= np.mean([x[-1] for x in test_err_ls])
        cur_loss = np.mean([x[-1] for x in test_loss_ls])
            #update best loss
        if cur_loss < min_loss:
            min_loss = cur_loss
            opt_iter = t
        ave_err.append(cur_err)
        ave_loss.append(cur_loss)
        
        # print report
        if t%20 == 0:
            print("%9.0f\t%13.4f\t%14.4f" % (t, cur_err,cur_loss))
            
        # detect early stop
        if t-opt_iter >= ESTOP: 
            print('Early stop criterion satisfied: break CV.')
            print('using iteration number:',opt_iter)
            break
    print("-----------------------------------------------\n")
    # visualization
    if plot:
        folder = inputs.output_folder
        if folder is None: 
            print("No CV file name provided.\n")
        else:
            f = plt.figure()
            plt.plot(ave_err)
            plt.xlabel("iterations")
            plt.ylabel("CV error")
            f.savefig(folder+'/CV_err.png')
            f=plt.figure()
            plt.plot(ave_loss)
            plt.xlabel("iterations")
            plt.ylabel("CV loss")
            f.savefig(folder+'/CV_loss.png')
    return opt_iter