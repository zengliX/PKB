# -*- coding: utf-8 -*-
"""
source file for input parameter processing
author: li zeng
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# test existence of data
def have_file(myfile):
    if not os.path.exists(myfile):
            print("file:",myfile,"does not exist")
            sys.exit(-1)
    else:
        print("file",myfile,"exists")

# input data class
class input_obj:
    "class object for input data"
    # data status
    loaded = False
    
    # input, output pars
    input_folder = None
    train_predictors=None
    train_response=None
    pred_sets=None
    output_folder=None

    # optional
    test_predictors = None
    test_file = None # name of test file

    # input summary
    Nsample = None
    Ngroup = None
    Ntrain = None
    Ntest = None
    group_names = None

    # model pars
    nu = None
    maxiter = 1000
    Lambda = None
    kernel = None
    method = None
    
    # ██████  ██████   ██████   ██████     ██ ███    ██ ██████  ██    ██ ████████
    # ██   ██ ██   ██ ██    ██ ██          ██ ████   ██ ██   ██ ██    ██    ██
    # ██████  ██████  ██    ██ ██          ██ ██ ██  ██ ██████  ██    ██    ██
    # ██      ██   ██ ██    ██ ██          ██ ██  ██ ██ ██      ██    ██    ██
    # ██      ██   ██  ██████   ██████     ██ ██   ████ ██       ██████     ██

    # function to process input
    def proc_input(self,config_file):
        """
        read configuration file
        load corresponding data
        """
        print('-------------- LOADING DATA ---------------')
        print("reading parameters from file:", config_file)
        # test file existence
        have_file(config_file)

        # input folder
        txt = open(config_file,"r")
        pars = yaml.load(txt)
        txt.close()

        # input folder
        if pars.get("input_folder"):
            config = pars.get('input_folder')
            self.input_folder = config
        else:
            print("Please specify input folder")
            sys.exit(-2)

        # output folder
        if pars.get("output_folder"):
            self.output_folder = config +"/"+ pars.get("output_folder")
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

        # training data
        if pars.get("predictor_set"):
            thisfile = config +"/"+ pars.get("predictor_set")
            have_file(thisfile)
            self.pred_sets = pd.Series.from_csv(thisfile)
        else:
            print("Group information missing.")
            sys.exit(-2)

        if pars.get("predictor"):
            thisfile = config + "/"+ pars.get("predictor")
            have_file(thisfile)
            self.train_predictors = pd.DataFrame.from_csv(thisfile)
        else:
            print("Training data: predictor missing")
            sys.exit(-2)

        if pars.get("response"):
            thisfile = config + "/"+ pars.get("response")
            have_file(thisfile)
            self.train_response = pd.DataFrame.from_csv(thisfile)
        else:
            print("Training data: response_train missing")
            sys.exit(-2)
        
        # testing data
        if pars.get("test_file"):
            thisfile = config + "/"+ pars.get("test_file")
            have_file(thisfile)
            self.test_predictors = pd.DataFrame.from_csv(thisfile)

        # load model parameters
        if pars.get("Lambda"):
            self.Lambda = float(pars.get("Lambda"))
        if pars.get("learning_rate"):
            self.nu = float(pars.get("learning_rate"))
        if pars.get("maxiter"):
            self.maxiter = pars.get("maxiter")
        if pars.get("kernel"):
            self.kernel = pars.get("kernel")
        if pars.get("method"):
            self.method= pars.get("method")
            
        # change loaded indicator
        self.loaded = True
        return
    
    def data_preprocessing(self,center = False):
        """
        preprocess data: 
        remove low variance, normalize each column, and drop not used groups
        """
        print('----------- DATA PRCOCESSING --------------')
        if not self.loaded: 
            print("No data loaded. Can not preprocess.")
            return
        
        # center data
        if center:
            print('Centering data...')
            scale(self.train_predictors,copy=False,with_std=False)
            if not self.test_predictors is None:
                scale(self.test_predictors,copy=False,with_std=False)
        
         # check groups
        print("Checking groups ...")
        to_drop =[]
        for i in range(len(self.pred_sets)):
            genes = self.pred_sets.values[i].split(" ")
            shared = np.intersect1d(self.train_predictors.columns.values,genes)
            if len(shared)==0:
                print("Drop group:",self.pred_sets.index[i])
                to_drop.append(i)
            else:
                self.pred_sets.values[i] = ' '.join(shared)              
        if len(to_drop)>0:        
            self.pred_sets = self.pred_sets.drop(self.pred_sets.index[to_drop])  
        
        # calculate summary
        self.Ntrain = len(self.train_response)
        self.Ngroup = len(self.pred_sets)
        self.Npred = self.train_predictors.shape[1]
        self.group_names = self.pred_sets.index
        if not self.test_predictors is None: 
            self.Ntest = self.test_predictors.shape[0]
        
        print('Done.')
        print('-------------------------------------------')
        return
        




    # ██████  ██████  ██ ███    ██ ████████
    # ██   ██ ██   ██ ██ ████   ██    ██
    # ██████  ██████  ██ ██ ██  ██    ██
    # ██      ██   ██ ██ ██  ██ ██    ██
    # ██      ██   ██ ██ ██   ████    ██

    def input_summary(self):
        print('------------ INPUT SUMMARY ----------------')
        print("input folder:", self.input_folder)
        print("output file folder:",self.output_folder)
        print("number of training samples:",self.Ntrain)
        print("number of testing samples:",self.Ntest)
        print("number of groups:", self.Ngroup)
        print("number of predictors:", self.Npred)
        return

    def model_param(self):
        print("learning rate:",self.nu)
        print("Lambda:",self.Lambda)
        print("maximum iteration:", self.maxiter)
        print("kernel function: ",self.kernel)
        print("method: ",self.method)
        print('-------------------------------------------')
        return
