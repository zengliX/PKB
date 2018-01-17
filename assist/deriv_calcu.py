"""
calculate derivatives
author: li zeng
"""
import numpy as np

# first order
def calcu_h(F_train,ytrain):
    denom  = np.exp(ytrain * F_train) + 1
    return (-ytrain)/denom


# second order
def calcu_q(F_train,ytrain):
    denom = (np.exp(ytrain * F_train) + 1)**2
    return np.exp(ytrain * F_train)/denom