#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:50:15 2019

@author: aliceliang
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LR import ar1_train
from MA import EMA_train
from KF import kf_train
from SVM import svm_train
#from PP import PP_train   # this model is too slow

import logging
logging.getLogger().setLevel(logging.ERROR)


"""
Change data directory before you run. The pd.read_excel() works differently
on different laptops (I guess it's a version problem) and please make sure 
it reads properly before running the entire program.
"""



"FACEBOOK"
# if this one does not work, uncomment next line
fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Open','High','Low','Close','Volume'], index_col="Date")
#fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, names=['Date','Open','High','Low','Close','Volume'], index_col=0)
fb.head(3)


"FANG"
# if this one does not work, uncomment next line
fang = pd.read_excel(io='~/Desktop/7374/FANG.xlsx', sheet_name='FANG', header=0, 
                     names=['AAPL','AMZN','FB','GOOGL','NFLX','NVDA','TSLA'], 
                     index_col="DATE", parse_dates=True)

#fang = pd.read_excel(io='~/Desktop/7374/FANG.xlsx', sheet_name='FANG', header=0, names=['DATE','AAPL','AMZN','FB','GOOGL','NFLX','NVDA','TSLA'], index_col=0, parse_dates=True)

fang_start = fang.index.get_loc('2016-01-06')       
fang_end = fang.index.get_loc('2018-08-31')
fang_slice=fang.iloc[fang_start:fang_end]
fang_fb=fang_slice.iloc[:,2]

"FF"
ff3 = pd.read_csv("/Users/aliceliang/Desktop/7374/hw3/FF_daily.csv", header=4,
                  skipfooter=2, index_col=0, engine='python', parse_dates=True)
ff3_start = ff3.index.get_loc('2016-01-04')
ff3_end = ff3.index.get_loc('2018-08-31')
ff3_fb=ff3.iloc[ff3_start:ff3_end]

"ADS"
ads = pd.read_excel(io="/Users/aliceliang/Desktop/7374/hw3/ADS_daily.xlsx", header=0, index_col=0)
ads_start = ads.index.get_loc('2016-01-04')
ads_end = ads.index.get_loc('2018-08-31')
ads_fb=ads.iloc[ads_start:ads_end]

"COMBINED"
result = pd.concat([fb, ff3_fb, ads_fb, fang_fb], axis = 1, 
                   join_axes=[fb.index])
result = result.fillna(0)



# df we are using 
df = result
df['O-C'] = df['Open']-df['Close']
df['H-L'] = df['High']-df['Low']
df['open_tmr'] = df['Open'].shift(-1) # tomorrow's open price
df['close_tmr'] = df['Close'].shift(-1) # tomorrow's open price
df['logVol'] = np.log(df['Volume']) # log Volume for scaling
df=df.dropna(0)






"global variables"

N = df.shape[0] # total num days
num_boot = 300 # total num bootstrap
T= 250 # start day
window = 200 # training period window


Y =df['open_tmr'][-(N-(T+1)):].values
Y_close = df['close_tmr'][-(N-(T+1)):].values

def get_Y():
    return Y, Y_close




"methods"


def ar1_bstr():
    yhat_ar1 = np.zeros(N-(T+1))
    for t in range(T+1, N):    
        # training data
        X_train = df[['Open','O-C','H-L','logVol']][t-window:t-1] # regression variables
        X_train = np.column_stack([np.ones((len(X_train),1)),X_train])
        Y_train = df['open_tmr'][t-window:t-1].values
        # one day prediction
        X_pred = df[['Open','O-C','H-L','logVol']][t-1:t]
        X_pred = np.column_stack([np.ones((len(X_pred),1)),X_pred])
        
        yhat_train = ar1_train(X_train, Y_train)[1]  
        res_train =  Y_train - yhat_train
        y_pred_all = np.zeros(num_boot)
        # bootstrap method: switching residuals
        for i in range(0, num_boot):
            err = np.random.choice(res_train, (window-1, ), replace=True)
            y_bstr = yhat_train + err   
            beta_bstr = ar1_train(X_train, y_bstr)[0]
            y_pred_bstr = X_pred@beta_bstr
            y_pred_all[i]=y_pred_bstr         
        y_pred_ar1 = y_pred_all.mean() # get mean of all bootstrap predictions        
        yhat_ar1[t-(T+1)]=y_pred_ar1 # do this for each time step
    rmse_ar1=np.sqrt(np.mean((Y-yhat_ar1)**2))
    return yhat_ar1, rmse_ar1

def ema_bstr():
    yhat_ema = np.zeros(N-(T+1))
    # 5 days EMA
    ema_short = df['Close'].ewm(span=5, adjust=False).mean()
    
    for t in range(T+1, N):
        X_train = ema_short[t-window:t-1]
        X_train = np.column_stack([np.ones((len(X_train),1)),X_train])        
        X_pred = ema_short[t-1:t]  
        X_pred = np.column_stack([np.ones((len(X_pred),1)),X_pred])        
        Y_train = df['open_tmr'][t-window:t-1].values
        yhat_fit = EMA_train(X_train,Y_train)[1] 
        res_fit = Y_train - yhat_fit
               
        # bootstrap method: switching residuals
        y_pred_all = np.zeros(num_boot)
        for i in range(0, num_boot):
            err = np.random.choice(res_fit, (window-1,), replace=True)
            y_bstr = yhat_fit + err           
            beta_bstr = EMA_train(X_train, y_bstr)[0]            
            y_pred_bstr = X_pred@beta_bstr
            y_pred_all[i]=y_pred_bstr            
        y_pred = y_pred_all.mean() # mean of all bootstrap predictions 
        yhat_ema[t-(T+1)]=y_pred        
    rmse_ema=np.sqrt(np.mean((Y-yhat_ema)**2))    
    return yhat_ema, rmse_ema


num_boot = 1
def pp_bstr():    
    yhat_pp = np.zeros(N-(T+1))   
    for t in range(T+1, N):     
        date = df.index[t-window:t-1].tolist()
        Y_train = df['open_tmr'][t-window:t-1].values 
        yhat_fit = PP_train(date, Y_train)[0] 
        res_fit = Y_train - yhat_fit
             
        # bootstrap method: switching residuals
        y_pred_all = np.zeros(num_boot)
        for i in range(0, num_boot):
            err = np.random.choice(res_fit, (window-1,), replace=True)
            y_bstr = yhat_fit + err             
            y_pred_bstr = PP_train(date, y_bstr)[1]          
            y_pred_all[i]=y_pred_bstr
            
        if t%30 ==0:
            print("prophet t:", t)  
            
        y_pred = y_pred_all.mean()      
        yhat_pp[t-(T+1)]=y_pred
    rmse_pp=np.sqrt(np.mean((Y-yhat_pp)**2))   
    return yhat_pp, rmse_pp



num_boot = 100
def kf_bstr():
    yhat_kf = np.zeros(N-(T+1))
    for t in range(T+1, N):    
        Y_train = df['open_tmr'][t-window:t-1].values
        yhat_fit = kf_train(Y_train)[0]
        res_fit = Y_train - yhat_fit
        
        # bootstrap method: switching residuals
        y_pred_all = np.zeros(num_boot)
        for i in range(0, num_boot):
            err = np.random.choice(res_fit, (window-1,), replace=True)
            y_bstr = yhat_fit + err                
            y_pred_bstr = kf_train(y_bstr)[1]
            y_pred_all[i]=y_pred_bstr            
        if t%100 ==0:
            print("kalman t:", t)    
        y_pred = y_pred_all.mean()
        yhat_kf[t-(T+1)]=y_pred
    
    rmse_kf=np.sqrt(np.mean((Y-yhat_kf)**2))
    return yhat_kf, rmse_kf



num_boot = 3 #  SVM is also very slow
def svm_bstr():
    yhat_svm = np.zeros(N-(T+1))
    for t in range(T+1, N):    
        X_train = df[['Close']][t-window:t-1] 
        X_train = np.column_stack([np.ones((len(X_train),1)),X_train])
        X_pred = df[['Close']][t-1:t]
        X_pred = np.column_stack([np.ones((len(X_pred),1)),X_pred])
        
        Y_train = df['open_tmr'][t-window:t-1].values    
        yhat_train = svm_train(X_train, Y_train, X_pred)[0] 
        res_train =  Y_train - yhat_train        
        y_pred_all = np.zeros(num_boot)
        # bootstrap method: switching residuals
        for i in range(0, num_boot):
            err = np.random.choice(res_train, (window-1, ), replace=True)
            y_bstr = yhat_train + err                        
            y_pred_bstr = svm_train(X_train, y_bstr, X_pred)[1]
            y_pred_all[i]=y_pred_bstr
            
        if t%100 ==0:
            print("svm t:", t)            
        y_pred_svm = y_pred_all.mean() # mean of all bootstrap predictions
        yhat_svm[t-(T+1)]=y_pred_svm # do this for each time step        
    rmse_svm=np.sqrt(np.mean((Y-yhat_svm)**2))
    return yhat_svm, rmse_svm