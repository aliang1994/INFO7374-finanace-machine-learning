#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:56:15 2019

@author: aliceliang
"""

import numpy as np
from numpy import log, pi, transpose, array
from numpy.linalg import inv, pinv
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt




def Kalman_Filter(param0, data):
    Y = np.asarray(data)
    #Y.shape = (671,1)
    
    T=Y.shape[0]

    # initial values
    #x_init = np.array([[y1[0]], [y2[0]]])
    x_init = np.array([[0.01], [0.01]])
    P_init = 900 * np.eye(len(x_init))  # small initial prediction error

    # 2x2 identity matrix
    Idn = np.eye(2)

    # allocate result matrices
    x_predict = np.zeros((T, 2, 1))      # prediction of state vector
    P_predict = np.zeros((T, 2, 2))  # prediction error covariance matrix
    x_update = np.zeros((T, 2, 1))       # estimation of state vector
    P_update = np.zeros((T, 2, 2))   # estimation error covariance matrix


    K = np.zeros((T, 2, 1))       # Kalman Gain
    v = np.zeros((T, 1, 1))  # error of estimate
    F = np.zeros((T, 1, 1))

    # initial x
    x_update[0] = x_init
    P_update[0] = P_init
    x_predict[0] = x_init
    P_predict[0] = P_init
    
    
    # params
    q= param0[0]
    r= param0[1]

    # matrices
    t = 1
    A = array([[1, t], [0, 1]])
    Q = q*array([[1, 0], [0, 1]])
    H = array([1, 0])
    H.shape = (1, 2)
    R = r*1

    #print("check: ", q, r)

    for i in range(T):
        # prediction stage
        if i > 0:
            x_predict[i] = A @ x_update[i-1]
            P_predict[i] = A @ P_update[i-1] @ transpose(A) + Q

        # estimation stage
        v[i] = Y[i] - H @ A @ x_update[i-1]
        F[i] = H @ P_predict[i] @ transpose(H) + R

        K[i] = P_predict[i] @ transpose(H) @ inv((H @ P_predict[i] @ transpose(H)) + R)
        
        x_update[i] = x_predict[i] + K[i] @ v[i] 
        P_update[i] = (Idn - K[i] @ H) @ P_predict[i]
    
        KF_Dens = A*log(2*pi) + log(abs(F[i])) + transpose(v[i]) @ inv(F[i]) @ v[i]
    
    return np.sum(KF_Dens)



def Kalman_Smoother(Y):
    param0 = array([1, 1])
    result = minimize(Kalman_Filter, param0, args=Y, method='BFGS', 
                  options={'gtol': 1e-8, 'disp': True})
    params_opt = result.x
    print(params_opt)
    
    
    qq= params_opt[0]
    rr= params_opt[1]
    
    T=Y.shape[0]
    
    # initial values
    #x_init = np.array([[y1[0]], [y2[0]]])
    x_init = np.array([[0.01], [0.01]])
    P_init = 900 * np.eye(len(x_init))  # small initial prediction error

    # 2x2 identity matrix
    Idn = np.eye(2)

    # allocate result matrices
    x_predict = np.zeros((T, 2, 1))      # prediction of state vector
    P_predict = np.zeros((T, 2, 2))  # prediction error covariance matrix
    x_update = np.zeros((T, 2, 1))       # estimation of state vector
    P_update = np.zeros((T, 2, 2))   # estimation error covariance matrix


    K = np.zeros((T, 2, 1))       # Kalman Gain
    v = np.zeros((T, 1, 1))  # error of estimate
    F = np.zeros((T, 1, 1))

    # initial x
    x_update[0] = x_init
    P_update[0] = P_init
    x_predict[0] = x_init
    P_predict[0] = P_init

    # matrices
    t = 1
    A = array([[1, t], [0, 1]])
    Q = qq*array([[1, 0], [0, 1]])
    H = array([1, 0])
    H.shape = (1, 2)
    R = rr*1


    for i in range(T):
        # prediction stage
        if i > 0:
            x_predict[i] = A @ x_update[i-1]
            P_predict[i] = A @ P_update[i-1] @ transpose(A) + Q

        # estimation stage
        v[i] = Y[i] - H @ A @ x_update[i-1]
        F[i] = H @ P_predict[i] @ transpose(H) + R

        K[i] = P_predict[i] @ transpose(H) @ inv((H @ P_predict[i] @ transpose(H)) + R)
        
        x_update[i] = x_predict[i] + K[i] @ v[i] 
        P_update[i] = (Idn - K[i] @ H) @ P_predict[i]
            
    "smoother"
    L = np.zeros((T,2,2))

    x_smooth = np.zeros((T, 2, 1))      # prediction of state vector
    P_smooth = np.zeros((T, 2, 2)) 

    x_smooth[T-1] = x_update[T-1]
    P_smooth[T-1] = P_update[T-1]  


    for t in range(T-1,0,-1): #range(start, stop, step)
        L[t-1] = P_update[t-1] @ transpose(A) @ pinv(P_predict[t-1]) 
        x_smooth[t-1] = x_update[t-1] + L[t-1] @ (x_smooth[t]-A @ x_update[t])
        P_smooth[t-1] = P_update[t-1] + L[t-1] @ (P_smooth[t]-P_update[t]) @ transpose(L[t-1])
    
    x_update.shape=(Y.shape[0],2,)
    x_smooth.shape=(Y.shape[0],2,)
  
    #print(x_smooth[0])
    return x_smooth, result.x


def Kalman_Filter_cv (Y, param0):
    T = Y.shape[0]
    
    # initial values
    #x_init = np.array([[y1[0]], [y2[0]]])
    x_init = np.array([[0.01], [0.01]])
    P_init = 900 * np.eye(len(x_init))  # small initial prediction error

    # 2x2 identity matrix
    Idn = np.eye(2)

    # allocate result matrices
    x_predict = np.zeros((T, 2, 1))      # prediction of state vector
    P_predict = np.zeros((T, 2, 2))  # prediction error covariance matrix
    x_update = np.zeros((T, 2, 1))       # estimation of state vector
    P_update = np.zeros((T, 2, 2))   # estimation error covariance matrix


    K = np.zeros((T, 2, 1))       # Kalman Gain
    v = np.zeros((T, 1, 1))  # error of estimate
    F = np.zeros((T, 1, 1))

    # initial x
    x_update[0] = x_init
    P_update[0] = P_init
    x_predict[0] = x_init
    P_predict[0] = P_init
    
    q= param0[0]
    r= param0[1]

    # matrices
    t = 1
    A = array([[1, t], [0, 1]])
    Q = q*array([[1, 0], [0, 1]])
    H = array([1, 0])
    H.shape = (1, 2)
    R = r*1

    for i in range(T):
        # prediction stage
        if i > 0:
            x_predict[i] = A @ x_update[i-1]
            P_predict[i] = A @ P_update[i-1] @ transpose(A) + Q

        # estimation stage
        v[i] = Y[i] - H @ A @ x_update[i-1]
        F[i] = H @ P_predict[i] @ transpose(H) + R

        K[i] = P_predict[i] @ transpose(H) @ inv((H @ P_predict[i] @ transpose(H)) + R)
        
        x_update[i] = x_predict[i] + K[i] @ v[i] 
        P_update[i] = (Idn - K[i] @ H) @ P_predict[i]
                
    "smoother"
    L = np.zeros((T,2,2))

    x_smooth = np.zeros((T, 2, 1))      # prediction of state vector
    P_smooth = np.zeros((T, 2, 2)) 

    x_smooth[T-1] = x_update[T-1]
    P_smooth[T-1] = P_update[T-1]  


    for t in range(T-1,0,-1): #range(start, stop, step)
        L[t-1] = P_update[t-1] @ transpose(A) @ pinv(P_predict[t-1]) 
        x_smooth[t-1] = x_update[t-1] + L[t-1] @ (x_smooth[t]-A @ x_update[t])
        P_smooth[t-1] = P_update[t-1] + L[t-1] @ (P_smooth[t]-P_update[t]) @ transpose(L[t-1])
    
    x_update.shape=(Y.shape[0],2,)
    x_smooth.shape=(Y.shape[0],2,)
  
    
    return x_smooth[:, 0]


"""
"running functions"

fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Open','High','Low','Close','Volume'], index_col='Date')

close = fb['Close']

T = close.shape[0]
Y = close

Y_smooth = Kalman_Smoother(close)


plt.figure(figsize=(40,20))
ax = plt.subplot(111)
timevec = np.linspace(1,T,T)
ax.plot(timevec, Y_smooth[:,0],'b', label = "K_smooth")
#ax.plot(timevec, Y_smooth,'b', label = "K_smooth")
ax.plot(timevec, Y, 'r', label = "actual price")
plt.title('Kalman Filter - Facebook')
ax.legend(loc=2, bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
plt.show()
"""
