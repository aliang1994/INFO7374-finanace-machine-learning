# -*- coding: utf-8 -*-
"""
Kalman Filter: Homework

@author: Yizhen Zhao
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import log, pi, transpose


def Kalman_Filter(Y):
    S = Y.shape[0]
    S = S + 1 # S=5
    
    "Initialize Params:"
    Z = param0[0] # 1.3
    T = param0[1]
    H = param0[2]
    Q = param0[3]
    
   
    "Kalman Filter Starts:"
    x_predict = np.zeros(S) # horizontal array of 0;
    P_predict = np.zeros(S)
    
    x_update = np.zeros(S)
    P_update = np.zeros(S)
    
    v = np.zeros(S)
    F = np.zeros(S)
    
    KF_Dens = np.zeros(S)
    
    for s in range(1,S): # index starts from 1
        if s == 1: 
            P_update[s] = 1000
            P_predict[s] = T*P_update[1]*transpose(T)+Q  
            
        else: 
            "Please fill this part."
            #predict
            x_predict[s]=T*x_predict[s-1]
            P_predict[s]=T*P_predict[s-1]*transpose(T)+Q
            
            #update
            v[s]=Y[s-1]-Z*x_predict[s]
            F[s]=Z*P_predict[s]*transpose(Z) + H          
            x_update[s]=x_predict[s]+P_predict[s]*transpose(Z)*(1/F[s])*v[s]
            P_update[s]=P_predict[s]-P_predict[s]*transpose(Z)*(1/F[s])*Z*P_predict[s]
            
            KF_Dens[s]= T*0.5*log(2*pi)+0.5*log(abs(F[s]))+0.5*v[s]*(1/F[s])*transpose(v[s])
                      
    Likelihood = np.sum(KF_Dens[1:-1])  
    return Likelihood


def Kalman_Smoother(params, Y):
    S = Y.shape[0]
    S = S + 1 # S=672

    "Initialize Params:"
    Z = params[0]
    T = params[1]
    H = params[2]
    Q = params[3]
 
    "Kalman Filter Starts:"
    x_predict = np.zeros(S)
    x_update = np.zeros(S)
    P_predict = np.zeros(S)
    P_update = np.zeros(S)
    
    v = np.zeros(S)
    F = np.zeros(S)
     
    for s in range(1,S):
        if s == 1: 
            P_update[s] = 1000
            P_predict[s] =  T*P_update[1]*transpose(T)+Q    
            x_predict[s] = 160
        else: 
            #predict
            x_predict[s]=T*x_predict[s-1]
            P_predict[s]=T*P_predict[s-1]*transpose(T)+Q
            
            #update
            v[s]=Y[s-1]-Z*x_predict[s]
            F[s]=Z*P_predict[s]*transpose(Z) + H          
            x_update[s]=x_predict[s]+P_predict[s]*transpose(Z)*(1/F[s])*v[s]
            P_update[s]=P_predict[s]-P_predict[s]*transpose(Z)*(1/F[s])*Z*P_predict[s]
            
    "Kalman Smoother:" 
    x_smooth = np.zeros(S)
    P_smooth = np.zeros(S)
    x_smooth[S-1] = x_update[S-1]
    P_smooth[S-1] = P_update[S-1]   
   

    
    for  t in range(S-1,0,-1): #range(start, stop, step)
        x_smooth[t-1] = x_update[t-1] + P_update[t-1]*transpose(T-1)*P_predict[t]*(x_predict[t]-T*x_update[t-1])
        P_smooth[t-1] = P_update[t-1] + P_update[t-1]*transpose(T-1)*(1/P_predict[t])*(P_smooth[t]-P_update[t])*transpose(P_update[t-1]*transpose(T-1)*(1/P_predict[t]))
        
    x_smooth = x_smooth[0:-1]
    x_update = x_update[1:S]
    
    return x_update, x_smooth


"Import Your Data here, and define as Y" 
"Remember to change initial values param0"

fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Date','Open','High','Low','Close','Volume'], 
                   index_col=None)

closep=fb['Close']
openp=fb['Open']
r = (closep-openp)/openp

T=fb.shape[0]
Y=r
timevec = np.linspace(1,T,T)
plt.figure(figsize=(20,10))



param0 = np.array([0.5, 0.3, 0.6, 0.8])
param_star = minimize(Kalman_Filter, param0, method='BFGS', 
                      options={'gtol': 1e-8, 'disp': True})

#Y_update = Kalman_Smoother(param_star.x, Y)[0]
Y_smooth = Kalman_Smoother(param_star.x, Y)[1]
timevec = np.linspace(1,T,T)
plt.plot(timevec, Y_smooth,'r',timevec, Y,'b')
