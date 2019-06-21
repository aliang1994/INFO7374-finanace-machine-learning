# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:43:49 2019
GARCH MODEL
@author: Yizhen Zhao
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from numpy import sqrt, log, pi


def GARCH(Y):
    "Initialize Params:"
    mu = param0[0]
    omega = param0[1]
    alpha = param0[2]
    beta = param0[3]
 
    T = Y.shape[0] 
    GARCH_Dens = np.zeros(T) 
    sigma2 = np.zeros(T) 
  
    #F = np.zeros(T)   
    #v = np.zeros(T)   
 
    for t in range(1,T): 
        # "Please fill this part."  
        
        "GARCH11"
        sigma2[t] = omega + alpha*(Y[t-1]-mu)**2 + beta*sigma2[t-1]
        GARCH_Dens[t] = 0.5*(log(2*pi) + log(sigma2[t]) + (Y[t]-mu)**2/sigma2[t])      
        Likelihood = np.sum(GARCH_Dens[1:-1])  
        return Likelihood


def GARCH_PROD(params, Y0, T):
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
 
    Y = np.zeros(T)  
    sigma2 = np.zeros(T)
    Y[0] = Y0
    sigma2[0] = 0.0001

    
    err=sp.random.standard_normal(T+1)
    for t in range(1,T):
        #"Please fill this part."          
        sigma2[t] = omega + alpha*(Y[t-1]-mu)**2 + beta*sigma2[t-1]
        Y[t] = mu + err[t]*sqrt(sigma2[t])        
    return Y    

# read FB closing price
fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Date','Open','High','Low','Close','Volume'], 
                   index_col=None)
closep=fb['Close']
openp=fb['Open']
"daily stock return"
rate = (closep-openp)/openp
T=fb.shape[0]
timevec = np.linspace(1,T,T)
plt.figure(figsize=(30,10))

Y2=rate*100 # percentage
param0 = np.array([0.1, 2.5, 0.2, 0.3])
param_star = minimize(GARCH, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})

Y2_GARCH = GARCH_PROD(param_star.x, Y2[0], T)
plt.plot(Y2,'b',Y2_GARCH,'r')