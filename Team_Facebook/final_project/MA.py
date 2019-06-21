# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:35:31 2019

@author: Tripti Santani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv



def EMA_train(X, Y):
    
    # OLS Method implementation yt=beta2*EMA(20)+uhatt        
    invXTX = np.linalg.inv(X.transpose()@X)
    beta_hat = invXTX@X.transpose()@Y
    
    y_hat = X@beta_hat
    
    
    """   
    (metrics commented out to save running time)
    
    T= Y.shape[0]        
    N = X.shape[1] 
    
    
    # residuals
    res = Y-y_hat
        
    # variance of error term / residuals
    var = (1/T)*res.transpose()@res  
    std = np.sqrt(var)
    #print(std)
    # variance-covariance matrix of beta_hat
    var_cov_beta = var*invXTX   
    std_cov_beta = np.sqrt(T*np.diag(var_cov_beta))
    
    # t-test statistics
    import scipy.stats as ss
    t_stat = (beta_hat.transpose()-0)/std_cov_beta
    p_val_t = 1-ss.norm.cdf(t_stat)
    #print("t_stat:", t_stat)
    #print("p_val_t:", p_val_t)
           
    # using MSE
    MSE = (sum((Y-y_hat)**2))/(len(X)-len(X[0]))
    var_b = MSE*(np.linalg.inv(np.dot(X.T,X)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = beta_hat/ sd_b
    from scipy import stats
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(X)-1))) for i in ts_b]
    #print("t_stat:", ts_b)
    #print("p_val_t:", p_values)
    
    #  F-test stats
    f_stat = (beta_hat.transpose()@inv(var_cov_beta)@beta_hat/N)/(res.transpose()@res/(T-N))
    p_val_f = 1-ss.f.cdf(f_stat, N-1, len(X)-N)
    
    #print("f_test: ", f_stat)
    #print("f_test p_val: ", p_val_f)
        
    # R-square
    R2 = 1-T*var/(T*np.var(Y))
    # R2 = 1 - (residuals.transpose()@residuals)/(Y.transpose()@Y)
    # ADJ R2
    R2_adj = 1-(1-R2)*(T-1)/(T-N)
    #print("EWMA model R2: ", R2, R2_adj)
    
    """
    
    return beta_hat, y_hat;