#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:58:17 2019


KALMAN FILTER


@author: aliceliang
"""
import numpy as np
from numpy import transpose, array
from numpy.linalg import inv


param0=[1.5, 1.5]

def kf_train(Y):
    Y = transpose(array(Y))
    T = Y.shape[0]
    
    # initial values
    x_init = np.array([[100], [0.01]]) # two states: stock price, percentage change in stock price
    P_init = 900 * np.eye(len(x_init))  

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

    x_smooth = np.zeros((T, 2, 1))      
    P_smooth = np.zeros((T, 2, 2)) 

    x_smooth[T-1] = x_update[T-1]
    P_smooth[T-1] = P_update[T-1]  
    
    
    for t in range(T-1,0,-1): #range(start, stop, step)
        L[t-1] = P_update[t-1] @ transpose(A) @ inv(P_predict[t-1]) 
        x_smooth[t-1] = x_update[t-1] + L[t-1] @ (x_smooth[t]-A @ x_update[t])
        P_smooth[t-1] = P_update[t-1] + L[t-1] @ (P_smooth[t]-P_update[t]) @ transpose(L[t-1])
     

    x_update.shape=(Y.shape[0],2,)
    x_smooth.shape=(Y.shape[0],2,)
    
    #one step forward prediction
    y_pred = A @ x_smooth[T-1]
    
    return x_smooth[:, 0], y_pred[0]