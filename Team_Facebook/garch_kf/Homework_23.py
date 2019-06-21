# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 09:04:31 2019

@author: Yizhen Zhao
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:43:49 2019
GARCH-t MODEL
@author: Yizhen Zhao
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy as sp
from numpy import sqrt, log, pi
import pandas as pd
from scipy.special import gamma


def GARCH_t(Y):
 "Initialize Params:"
 mu = param0[0]
 omega = param0[1]
 alpha = param0[2]
 beta = param0[3]
 nv = param0[4]
 
 T = Y.shape[0]
 GARCH_t = np.zeros(T) 
 sigma2 = np.zeros(T)   
 #F = np.zeros(T)   
 #v = np.zeros(T)   
 for t in range(1,T):
    #"Please fill this part. "
    "GARCH11"
    sigma2[t]=omega + alpha*(Y[t-1]-mu)**2 + beta*sigma2[t-1]
    #GARCH_t[t]=0.5*(log(2*pi) + log(sigma2[t]) + (Y[t]-mu)**2/sigma2[t])
    GARCH_t[t]=-log(gamma((nv+1)/2)/sqrt(pi*(nv-2))*gamma(nv/2))+0.5*log(sigma2[t])+((nv+1)/2)*log(1+(Y[t]-mu)**2/sigma2[t]*(nv-2))
    Likelihood=np.sum(GARCH_t[1:-1])  
 
 return Likelihood


def GARCH_PROD_t(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 nv = params[4]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 
 err = np.random.standard_t(nv, size = T)
 
 Y[0] = Y0
 sigma2[0] = 0.00001
 for t in range(1,T):
   #"Please fill this part."
   sigma2[t] = omega + alpha*(Y[t-1]-mu)**2 + beta*sigma2[t-1]
   Y[t] = mu + err[t]*sqrt(sigma2[t])
 return Y    


"Please import your data here and change initial values of param0."
# read FB closing price
fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Date','Open','High','Low','Close','Volume'], 
                   index_col=None)

closep=fb['Close']
openp=fb['Open']

"daily stock return"
rate_t = (closep-openp)/openp

T=fb.shape[0]
Y_t=rate_t*100 # percentage
timevec = np.linspace(1,T,T)
plt.figure(figsize=(30,10))

param0 = np.array([0.1, 2.5, 0.2, 0.3, 5])
param_star = minimize(GARCH_t, param0, method='BFGS', options={'gtol': 1e-8, 'disp': True})
Y_GARCH_t = GARCH_PROD_t(param_star.x, Y_t[0], T)
timevec = np.linspace(1,T,T)
plt.plot(timevec, Y_t,'b',timevec, Y_GARCH_t,'r')