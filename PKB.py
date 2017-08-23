# -*- coding: utf-8 -*-
"""
main program
author: li zeng
"""

# ██ ███    ███ ██████   ██████  ██████  ████████
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██
# ██ ██      ██ ██       ██████  ██   ██    ██


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import assist
from assist.method_L1 import oneiter_L1, find_Lambda_L1
from assist.method_L2 import oneiter_L2, find_Lambda_L2
import numpy as np
import time
from multiprocessing import cpu_count
from sys import argv
import sharedmem
import scipy
import pandas as pd

"""-----------------
FUNCTIONS
--------------------"""
# loss function
def loss_fun(f,y):
    return np.mean(np.log(1+np.exp(-y*f)))    

# line search
def line_search(sharedK,F_train,ytrain,Kdims,pars):
    [m,beta,c] = pars    
    # get K
    nrow = Kdims[0]
    width = nrow**2
    Km = sharedK[(m*width):((m+1)*width)].reshape((nrow,nrow))
    
    b = Km.dot(beta)+c
    # line search function
    def temp_fun(x):
        return np.log(1+np.exp(-ytrain*(F_train+ x*b))).sum()
    out = scipy.optimize.minimize_scalar(temp_fun)
    if not out.success:
        print("warning: minimization failure")
    return out.x    

# ██ ███    ██ ██████  ██    ██ ████████
# ██ ████   ██ ██   ██ ██    ██    ██
# ██ ██ ██  ██ ██████  ██    ██    ██
# ██ ██  ██ ██ ██      ██    ██    ██
# ██ ██   ████ ██       ██████     ██

# set random seed
np.random.seed(1)

# input:  folder with data and configuration file
#config_file = './example/config_file.txt'
config_file = argv[1]
print("input file:",config_file)
inputs = assist.input_process.input_obj()
inputs.proc_input(config_file)
inputs.data_preprocessing(rm_lowvar=False,center=True,norm=False)


"""---------------------------
SPLIT TEST TRAIN
----------------------------"""
inputs.data_split()
# input summary
print()
# inputs.data_split()
inputs.input_summary()
inputs.model_param()
print("number of cpus available:",cpu_count())
print()


# ██████  ██    ██ ███    ██      █████  ██       ██████   ██████
# ██   ██ ██    ██ ████   ██     ██   ██ ██      ██       ██    ██
# ██████  ██    ██ ██ ██  ██     ███████ ██      ██   ███ ██    ██
# ██   ██ ██    ██ ██  ██ ██     ██   ██ ██      ██    ██ ██    ██
# ██   ██  ██████  ██   ████     ██   ██ ███████  ██████   ██████

"""---------------------------
CALCULATE KERNEL
----------------------------"""
K_train = assist.kernel_calc.get_kernels(inputs.train_predictors,inputs.train_predictors,inputs)
if inputs.Ntest > 0:
    K_test= assist.kernel_calc.get_kernels(inputs.train_predictors,inputs.test_predictors,inputs)

# put K_train in shared memory
sharedK = sharedmem.empty(K_train.size)
sharedK[:] = np.transpose(K_train,(2,0,1)).reshape((K_train.size,))
Kdims = (K_train.shape[0],K_train.shape[2])

"""---------------------------
initialize outputs object
----------------------------"""
outputs = assist.outputs.output_obj(inputs)

ytrain = np.squeeze(inputs.train_response.values)
N1 = (ytrain == 1).sum()
N0 = (ytrain ==-1).sum()
F0 = np.log(N1/N0) # initial value
if F0==0: F0+=10**(-2)
outputs.F0 = F0

F_train = np.repeat(F0,inputs.Ntrain) # keep track of F_t(x_i) on training data
outputs.train_err.append((np.sign(F_train) != ytrain).sum()/len(ytrain))
outputs.train_loss.append(loss_fun(F_train,ytrain))
if inputs.Ntest > 0:
    ytest = np.squeeze(inputs.test_response.values)
    F_test = np.repeat(F0,inputs.Ntest) # keep track of F_t(x_i) on testing data
    outputs.test_err.append((np.sign(F_test) != ytest).sum()/len(ytest))
    outputs.test_loss.append(loss_fun(F_test,ytest))
    
outputs.trace.append([0,np.repeat(0,inputs.Ntrain),0])


# ██████   ██████   ██████  ███████ ████████ ██ ███    ██  ██████
# ██   ██ ██    ██ ██    ██ ██         ██    ██ ████   ██ ██
# ██████  ██    ██ ██    ██ ███████    ██    ██ ██ ██  ██ ██   ███
# ██   ██ ██    ██ ██    ██      ██    ██    ██ ██  ██ ██ ██    ██
# ██████   ██████   ██████  ███████    ██    ██ ██   ████  ██████
"""---------------------------
BOOSTING PARAMETERS
----------------------------"""
ncpu = min(5,cpu_count())
Lambda = inputs.Lambda

# automatic selection of Lambda
if inputs.method == 'L1' and Lambda is None: 
    Lambda = find_Lambda_L1(K_train,F_train,ytrain,Kdims)
    print("L1 method: use Lambda",Lambda)
if inputs.method == 'L2' and Lambda is None:
    Lambda = find_Lambda_L2(K_train,F_train,ytrain,Kdims,C=1)
    print("L2 method: use Lambda",Lambda)

# is there a need to run parallel
if (inputs.Ntrain > 500 or inputs.Ngroup > 40):
    parallel = True
    print("Algorithm: parallel on",ncpu,"cores")
    gr_sub = True
    print("Algorithm: random groups selected in each iteration")
else:
    parallel = False
    gr_sub = False
    print("Algorithm: parallel algorithm not used")

ESTOP = 500 # early stop if test_loss have no increase
if inputs.Ntest>0: loss0 = outputs.test_loss[0]
ct_fromMinLoss = 0 # count from minLoss

"""---------------------------
BOOSTING ITERATIONS
----------------------------"""

time0 = time.time()
for t in range(1,inputs.maxiter+1):
    # one iteration
    if inputs.method == 'L2':
        [m,beta,c] = oneiter_L2(sharedK,F_train,ytrain,Kdims,\
                Lambda=Lambda,ncpu = ncpu,parallel = parallel,subset=inputs.rand_subset)    
    if inputs.method == 'L1':
        [m,beta,c] = oneiter_L1(sharedK,F_train,ytrain,Kdims,\
                Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                subset=inputs.rand_subset,group_subset = gr_sub) 
    
    # line search
    x = line_search(sharedK,F_train,ytrain,Kdims,[m,beta,c])
    beta *= x
    c *= x
    
    # update outputs
    outputs.trace.append([m,beta,c])
    F_train += (K_train[:,:,m].dot(beta) + c)*inputs.nu
    outputs.train_err.append((np.sign(F_train)!=ytrain).sum()/len(ytrain))
    outputs.train_loss.append(loss_fun(F_train,ytrain))
    
    if inputs.Ntest>0:
        F_test += (K_test[:,:,m].T.dot(beta)+ c)*inputs.nu
        outputs.test_err.append((np.sign(F_test)!=ytest).sum()/len(ytest))
        new_loss = loss_fun(F_test,ytest)
        outputs.test_loss.append(loss_fun(F_test,ytest))
        if new_loss < loss0:
            ct_fromMinLoss = 0; loss0 = new_loss
        else:
            ct_fromMinLoss += 1
        # check early stop 
        if ct_fromMinLoss == ESTOP: 
            print('Early stop criterion satisfied: break boosting.\n')
            break
                   
    # print time report
    if t%50 == 0:
        print("iteration:",t)
        print("training error:",outputs.train_err[t])
        if inputs.Ntest > 0: print("testing error:",outputs.test_err[t])
        iter_persec = t/(time.time() - time0) # time of one iteration
        print("Speed:",iter_persec,"iterations/second")
        rem_time = (inputs.maxiter-t)/iter_persec # remaining time
        print("Time remaining:",rem_time/60,"mins")
        print()



# ██████  ███████ ███████ ██    ██ ██   ████████ ███████
# ██   ██ ██      ██      ██    ██ ██      ██    ██
# ██████  █████   ███████ ██    ██ ██      ██    ███████
# ██   ██ ██           ██ ██    ██ ██      ██         ██
# ██   ██ ███████ ███████  ██████  ███████ ██    ███████

## show results
opt_t = outputs.best_t()

    # trace
f = outputs.show_err()
f.savefig(inputs.output_folder + "/err.pdf")

    # opt weights
[weights,f0] = outputs.group_weights(opt_t,plot=True)
f0.savefig(inputs.output_folder + "/opt_weights.pdf")

    # opt weights list
sorted_w = pd.Series(weights,index=inputs.group_names).sort_values(ascending=False)
sorted_w.to_csv(inputs.output_folder+'/opt_weights.txt',index_label='group')

    # weights paths
[path_mat,f1,f2] = outputs.weights_path(plot=True)
f1.savefig(inputs.output_folder + "/weights_path1.pdf")
f2.savefig(inputs.output_folder + "/weights_path2.pdf")


# ███████  █████  ██    ██ ███████
# ██      ██   ██ ██    ██ ██
# ███████ ███████ ██    ██ █████
#      ██ ██   ██  ██  ██  ██
# ███████ ██   ██   ████   ███████

# save outputs to files
import pickle
out_file = inputs.output_folder + "/results.pckl"
f = open(out_file,'wb')
outputs.inputs.test_predictors =None
outputs.inputs.train_predictors = None
outputs.inputs.test_response=None
outputs.inputs.train_response=None
outputs.inputs.pred_sets=None
pickle.dump(outputs,f)
f.close()
print("results saved to:",out_file)


