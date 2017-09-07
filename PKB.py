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
from assist.functions import loss_fun, line_search
from assist.cvPKB import CV_PKB
from assist.method_L1 import oneiter_L1, find_Lambda_L1
from assist.method_L2 import oneiter_L2, find_Lambda_L2
import numpy as np
import pandas as pd
import time
from multiprocessing import cpu_count
from sys import argv
import sharedmem
  

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
inputs.data_preprocessing(center=True)


"""---------------------------
SPLIT TEST TRAIN
----------------------------"""
# input summary
print()
inputs.input_summary()
inputs.model_param()
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
if not inputs.Ntest is None:
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
if not inputs.Ntest is None:
    F_test = np.repeat(F0,inputs.Ntest) # keep track of F_t(x_i) on testing data
    
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
    print("Algorithm: parallel algorithm not used\n")

ESTOP = 100 # early stop if test_loss have no increase

"""---------------------------
CV FOR NUMBER OF ITERATIONS
----------------------------"""
opt_iter = CV_PKB(inputs,sharedK,K_train,Kdims,Lambda,nfold=3,ESTOP=ESTOP,\
                  ncpu=1,parallel=False,gr_sub=False,plot=True)


"""---------------------------
BOOSTING ITERATIONS
----------------------------"""
time0 = time.time()
print("--------------------- Boosting -------------------")
print("iteration\ttrain err\t time(min)")
for t in range(1,opt_iter+1):
    # one iteration
    if inputs.method == 'L2':
        [m,beta,c] = oneiter_L2(sharedK,F_train,ytrain,Kdims,\
                Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                sele_loc=None,group_subset = gr_sub)    
    if inputs.method == 'L1':
        [m,beta,c] = oneiter_L1(sharedK,F_train,ytrain,Kdims,\
                Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                sele_loc=None,group_subset = gr_sub) 
    
    # line search
    x = line_search(sharedK,F_train,ytrain,Kdims,[m,beta,c])
    beta *= x
    c *= x
    
    # update outputs
    outputs.trace.append([m,beta,c])
    F_train += (K_train[:,:,m].dot(beta) + c)*inputs.nu
    outputs.train_err.append((np.sign(F_train)!=ytrain).sum()/len(ytrain))
    outputs.train_loss.append(loss_fun(F_train,ytrain))
    if not inputs.Ntest is None:
        F_test += (K_test[:,:,m].T.dot(beta)+ c)*inputs.nu
                   
    # print time report
    if t%20 == 0:
        iter_persec = t/(time.time() - time0) # time of one iteration
        rem_time = (opt_iter-t)/iter_persec # remaining time
        print("%9.0f\t%9.4f\t%9.4f" % \
              (t,outputs.train_err[t],rem_time/60))
print("--------------------------------------------------")



# ██████  ███████ ███████ ██    ██ ██   ████████ ███████
# ██   ██ ██      ██      ██    ██ ██      ██    ██
# ██████  █████   ███████ ██    ██ ██      ██    ███████
# ██   ██ ██           ██ ██    ██ ██      ██         ██
# ██   ██ ███████ ███████  ██████  ███████ ██    ███████
    # predictions
if not inputs.Ntest is None:
    pred = pd.Series( np.array([int(x>0) for x in F_test])*2-1, index= inputs.test_predictors.index )
    pred.to_csv(inputs.output_folder+"/test_prediction.txt",index_label='sample')
    
    # opt weights
[weights,f0] = outputs.group_weights(opt_iter,plot=True)
f0.savefig(inputs.output_folder + "/opt_weights.png")

    # opt weights list
sorted_w = pd.Series(weights,index=inputs.group_names).sort_values(ascending=False)
sorted_w.to_csv(inputs.output_folder+'/opt_weights.txt',index_label='group')

    # weights paths
[path_mat,f] = outputs.weights_path(plot=True)
f.savefig(inputs.output_folder + "/weights_path.png")


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


