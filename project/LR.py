#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:31:47 2019

@author: aliceliang

linear regression
"""

import numpy as np
import scipy.stats as ss
from numpy.linalg import inv



def ar1_train(X, Y):
    invXTX = np.linalg.inv(X.transpose()@X)
    beta_hat = invXTX@X.transpose()@Y    
    y_hat = X@beta_hat
    
    
    """
    commented out to save bootstrap running time......
    
    T = Y.shape[0]
    N = X.shape[1]  
    
    # variance of residuals
    var = (1/T)*res.transpose()@res
    std = np.sqrt(var)
    
    # variance-covariance matrix of beta_hat
    var_cov_beta = var*invXTX
    std_cov_beta = np.sqrt(T*np.diag(var_cov_beta))
    
    # t-test 
    from scipy import stats
    mse = (sum((Y-y_hat)**2))/(len(X)-N)
    var_beta = mse*invXTX.diagonal()
    sd_beta = np.sqrt(var_beta)
    ts_beta = beta_hat/sd_beta
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(X)-1))) for i in ts_beta]
        
    # F-test 
    f_stat = (beta_hat.transpose()@inv(var_cov_beta)@beta_hat/N)/(res.transpose()@res/(T-N))
    p_val_f = 1-ss.f.cdf(f_stat, N-1, len(X)-N)
    
    # R-square
    R2 = 1-T*var/(T*np.var(Y))
    R2_adj = 1-(1-R2)*(T-1)/(T-N)

    """
    return beta_hat, y_hat;