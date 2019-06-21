#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:31:47 2019

@author: aliceliang

linear regression
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

from numpy import pi, log, transpose, array
from numpy.linalg import inv



# result is the df


def ar1_train(df, num_days):
    T = df.shape[0]
    Y=df['lag'][:num_days].values
    X=df[['Open','O-C','H-L','Volume','ADS_Index']][:num_days] # regression variables
    X = np.column_stack([np.ones((len(X),1)),X])
    N = X.shape[1]
    #print("Y shape: ", Y.shape, "X shape", X.shape)
    
    # inverse(X'X)
    invXTX = np.linalg.inv(X.transpose()@X)
    # beta_hat = inverse(X'X)(X'Y)
    beta_hat = invXTX@X.transpose()@Y
    #print("beta_hat: ", beta_hat)
    
    # y_hat, residuals
    y_hat = X@beta_hat
    res = Y-y_hat
    
    # variance of residuals
    var = (1/T)*res.transpose()@res
    std = np.sqrt(var)
    
    # variance-covariance matrix of beta_hat
    var_cov_beta = var*invXTX
    std_cov_beta = np.sqrt(T*np.diag(var_cov_beta))
    
    # t-test stats
    from scipy import stats
    mse = (sum((Y-y_hat)**2))/(len(X)-N)
    var_beta = mse*invXTX.diagonal()
    sd_beta = np.sqrt(var_beta)
    ts_beta = beta_hat/sd_beta
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(X)-1))) for i in ts_beta]
    
    #print("ar1 t_test: ", ts_beta)
    #print("ar1 t_test p_val: ", p_values)
    
    #  F-test stats
    f_stat = (beta_hat.transpose()@inv(var_cov_beta)@beta_hat/N)/(res.transpose()@res/(T-N))
    p_val_f = 1-ss.f.cdf(f_stat, N-1, len(X)-N)
    
    #print("ar1 f_test: ", f_stat)
    #print("ar1 f_test p_val: ", p_val_f)
    
    # R-square
    R2 = 1-T*var/(T*np.var(Y))
    R2_adj = 1-(1-R2)*(T-1)/(T-N)
    #print("ar1 R2 and adj_R2: ", R2, R2_adj)
    
    return beta_hat, res, y_hat;


def ar1_cv(y_bstr, df, num_days):
    T = df.shape[0]
    Y= y_bstr[:num_days]
    X=df[['Open','O-C','H-L','Volume','ADS_Index']][:num_days] # regression variables
    X = np.column_stack([np.ones((len(X),1)),X])
    N = X.shape[1]
    #print("Y shape: ", Y.shape, "X shape", X.shape)
    
    # inverse(X'X)
    invXTX = np.linalg.inv(X.transpose()@X)
    # beta_hat = inverse(X'X)(X'Y)
    beta_hat = invXTX@X.transpose()@Y
    #print("beta_hat: ", beta_hat)
    
    # y_hat, residuals
    y_hat = X@beta_hat
    res = Y-y_hat
    
    # variance of residuals
    var = (1/T)*res.transpose()@res
    std = np.sqrt(var)
    
    # variance-covariance matrix of beta_hat
    var_cov_beta = var*invXTX
    std_cov_beta = np.sqrt(T*np.diag(var_cov_beta))
    
    # t-test stats
    from scipy import stats
    mse = (sum((Y-y_hat)**2))/(len(X)-N)
    var_beta = mse*invXTX.diagonal()
    sd_beta = np.sqrt(var_beta)
    ts_beta = beta_hat/sd_beta
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(X)-1))) for i in ts_beta]
    
    #print("ar1 t_test: ", ts_beta)
    #print("ar1 t_test p_val: ", p_values)
    
    #  F-test stats
    f_stat = (beta_hat.transpose()@inv(var_cov_beta)@beta_hat/N)/(res.transpose()@res/(T-N))
    p_val_f = 1-ss.f.cdf(f_stat, N-1, len(X)-N)
    
    #print("ar1 f_test: ", f_stat)
    #print("ar1 f_test p_val: ", p_val_f)
    
    # R-square
    R2 = 1-T*var/(T*np.var(Y))
    R2_adj = 1-(1-R2)*(T-1)/(T-N)
    #print("ar1 R2 and adj_R2: ", R2, R2_adj)
    
    # prediction
    # one day prediction
    
    
    return beta_hat, res, y_hat;


