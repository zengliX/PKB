# -*- coding: utf-8 -*-
"""
The outputs class
author: li zeng
"""

import numpy as np
from matplotlib import pyplot as plt

def weight_calc(mat):
    weights = []
    for j in range(mat.shape[1]):
        weights.append( np.sqrt((mat[:,j]**2).sum()) )
    return weights

class output_obj:
    """Object for outputs"""        
    # initialization
    def __init__(self,inputs):
        self.inputs = inputs
        self.coef_mat =  np.zeros([inputs.Ntrain,inputs.Ngroup])
        self.F0 = None
        
        # tracking of performance
        self.trace = []  # keep track of each iteration
        self.train_err = [] # training error
        self.test_err = [] # testing error    
        self.train_loss = [] # loss function at each iteration
        self.test_loss = []  
        return

    # ██████   ██████  ███████ ████████      █████  ██       ██████   ██████
    # ██   ██ ██    ██ ██         ██        ██   ██ ██      ██       ██    ██
    # ██████  ██    ██ ███████    ██        ███████ ██      ██   ███ ██    ██
    # ██      ██    ██      ██    ██        ██   ██ ██      ██    ██ ██    ██
    # ██       ██████  ███████    ██        ██   ██ ███████  ██████   ██████
    # methods to be used after boosting
    
    # show the trace of fitting
    def show_err(self):
        f = plt.figure()
        plt.plot(self.train_err,'b')
        plt.plot(self.test_err,'r')
        plt.xlabel("iterations")
        plt.ylabel("classification error")
        plt.text(len(self.train_err),self.train_err[-1], "training error")
        plt.text(len(self.test_err),self.test_err[-1], "testing error")
        plt.title("Classifiction errors in each iteration")
        return f
    
    def show_loss(self):
        f = plt.figure()
        plt.plot(self.train_loss,'b')
        plt.plot(self.test_loss,'r')
        plt.xlabel("iterations")
        plt.ylabel("loss function")
        plt.text(len(self.train_loss),self.train_loss[-1], "training loss")
        plt.text(len(self.test_loss),self.test_loss[-1], "testing loss")
        plt.title("Loss function at each iteration")
        return f
    
    def group_weights(self,t,plot=True):
        self.coef_mat.fill(0)
        # calculate coefficient matrix at step t
        for i in range(t+1):
            [m,beta,c] = self.trace[i]
            self.coef_mat[:,m] += beta*self.inputs.nu
                         
        # calculate pathway weights
        weights = weight_calc(self.coef_mat)
        
        # visualization
        if plot:
            f=plt.figure()
            plt.bar(range(1,self.inputs.Ngroup+1),weights)
            plt.xlabel("groups")
            plt.ylabel("group weights")

        return [weights,f]
    
    # show the path of weights for each group
    def weights_path(self,plot=True):
        self.coef_mat.fill(0)
        # calculate coefficient matrix at step t
        weight_mat = np.zeros([len(self.train_err),self.inputs.Ngroup])
        
        # calculate weights for each iteration
        for i in range(1,len(self.train_err)):
            #[m,n,alpha,c,beta] = self.trace[i]
            #self.coef_mat[n,m] += self.inputs.nu*beta*alpha
            [m,beta,c] = self.trace[i]
            self.coef_mat[:,m] += beta*self.inputs.nu
            weight_mat[i,:] = weight_mat[i-1,:]
            weight_mat[i,m] =  np.sqrt((self.coef_mat[:,m]**2).sum())
        
        # weights at opt_t     
        first5 = weight_mat[-1,:].argsort()[-5:][::-1]
        
        # visualization
        if plot:
            f1=plt.figure()
            for m in range(weight_mat.shape[1]):
                if m in first5:
                    plt.plot(weight_mat[:,m],label=str(self.inputs.group_names[m]))
                else:
                    plt.plot(weight_mat[:,m])
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("weights")
            plt.title("group weights dynamics")
        return [weight_mat,f1]
       
        
        
        