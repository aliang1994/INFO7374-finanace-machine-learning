#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:50:15 2019

@author: aliceliang
"""
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import logging

logging.getLogger().setLevel(logging.WARNING)
logging.propagate = False

"1.Read and combine data frames"

"FACEBOOK"
fb = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Open','High','Low','Close','Volume'], index_col='Date')
"FANG"
fang = pd.read_excel(io='~/Desktop/7374/FANG.xlsx', sheet_name='FANG', header=0, 
                     names=['AAPL','AMZN','FB','GOOGL','NFLX','NVDA','TSLA'], 
                     index_col='DATE',parse_dates=True)
fang_start = fang.index.get_loc('2016-01-06')
fang_end = fang.index.get_loc('2018-08-31')
fang_slice=fang.iloc[fang_start:fang_end]
fang_fb=fang_slice.iloc[:,2]

"FF"
ff3 = pd.read_csv("/Users/aliceliang/Desktop/7374/hw3/FF_daily.csv", header=4,
                  skipfooter=2, index_col=0,engine='python', parse_dates=True)
ff3_start = ff3.index.get_loc('2016-01-04')
ff3_end = ff3.index.get_loc('2018-08-31')
ff3_fb=ff3.iloc[ff3_start:ff3_end]

"ADS"
ads = pd.read_excel(io="/Users/aliceliang/Desktop/7374/hw3/ADS_daily.xlsx", 
                    header=0, index_col=0)
ads_start = ads.index.get_loc('2016-01-04')
ads_end = ads.index.get_loc('2018-08-31')
ads_fb=ads.iloc[ads_start:ads_end]

"COMBINED"
result = pd.concat([fb, ff3_fb, ads_fb, fang_fb], axis = 1, 
                   join_axes=[fb.index])
result = result.fillna(0)
# result.to_csv('mergeresult.csv')



"2. Model Performance"

df = result
df['O-C'] = df['Open']-df['Close']
df['H-L'] = df['High']-df['Low']
df['lag'] = df['Open'].shift(-1)
df=df.dropna(0)

N = df.shape[0] # total days
num_boot = 200 # number of times for bootstrap
T= 210 # start day
window = 200 # training days

print("---------------------Model 1: Enhanced AR(1)--------------------")
from LR import ar1_train, ar1_cv
# training
# beta_ar1 = ar1_train(df,T)[0]
# res_ar1 = ar1_train(df,T)[1]
# yhat_ar1 = ar1_train(df,T)[2]
#print("shape: ",yhat_ar1.shape)


# crossvalidation
# Y = df['lag']

Y_cv_ar1 =df['lag'][-(N-(T+1)):].values
yhat_cv_ar1 = np.zeros(N-(T+1))

for t in range(T+1, N):
    # fit
    df_cv = df[t-window:t-1]

    res_fit = ar1_train(df_cv, window-1)[1]
    yhat_fit = ar1_train(df_cv, window-1)[2]    
    
    # pred data
    df_pred = df[t-1:t]
    X_pred = df_pred[['Open','O-C','H-L','Volume','ADS_Index']]
    X_pred = np.column_stack([np.ones((len(X_pred),1)),X_pred])
    
    # bootstrap method: switching residuals
    y_pred_all = np.zeros(num_boot)
    for i in range(0, num_boot):
        err = np.random.choice(res_fit, (window-1,), replace=True)
        y_bstr = yhat_fit + err   
        beta_bstr = ar1_cv(y_bstr, df_cv, window-1)[0]
        y_pred_bstr = X_pred@beta_bstr
        y_pred_all[i]=y_pred_bstr
        
    #print("bootstrap fininished for t:", t)    
    y_pred_ar1 = y_pred_all.mean()
    yhat_cv_ar1[t-T-1]=y_pred_ar1
    

rmse_ar1=np.sqrt(np.mean((Y_cv_ar1-yhat_cv_ar1)**2))
print("ar1_RMSE_cv: ", rmse_ar1)


print("---------------------Model 2: EWMA----------------------------")
from MA import EMA_train, EMA_cv


# crossvalidation
Y_cv_ema =df['lag'][-(N-(T+1)):].values
yhat_cv_ema = np.zeros(N-(T+1))

for t in range(T+1, N):
    # fit
    df_cv = df[t-window:t-1]

    res_fit = EMA_train(df_cv)[1]
    yhat_fit = EMA_train(df_cv)[2] 
    ema_short = EMA_train(df_cv)[3] 
    
    # pred data 
    X_pred = ema_short[-1:]
    X_pred = np.column_stack([np.ones((len(X_pred),1)),X_pred])
    
    # bootstrap method: switching residuals
    y_pred_all = np.zeros(num_boot)
    for i in range(0, num_boot):
        err = np.random.choice(res_fit, (window-1,), replace=True)
        y_bstr = yhat_fit + err   
        
        beta_bstr = EMA_cv(y_bstr, df_cv)[0]
        
        y_pred_bstr = X_pred@beta_bstr
        y_pred_all[i]=y_pred_bstr
        
    #print("bootstrap fininished for t:", t)    
    y_pred = y_pred_all.mean()
    
    yhat_cv_ema[t-T-1]=y_pred
    

rmse_ema=np.sqrt(np.mean((Y_cv_ema-yhat_cv_ema)**2))
print("ema_RMSE_cv: ", rmse_ema)

"""
print("---------------------Model 3: FB Prophet----------------------------")
from Prophet import PP_train, PP_cv

# crossvalidation
Y_cv_pp =df['Open'][-(N-(T+1)):].values
yhat_cv_pp = np.zeros(N-(T+1))

for t in range(T+1, N):
    # fit
    df_cv = df[t-window:t-1]

    res_fit = PP_train(df_cv)[0]
    #print("res_fit: ",res_fit)
    yhat_fit = PP_train(df_cv)[1] 
    #print("yhat_fit: ",yhat_fit)

    # pred data 
    df_pred = df[t-1:t]
    X_pred = ema_short[-1:]
    X_pred = np.column_stack([np.ones((len(X_pred),1)),X_pred])
    
    # bootstrap method: switching residuals
    y_pred_all = np.zeros(num_boot)
    for i in range(0, num_boot):
        err = np.random.choice(res_fit, (window-1,), replace=True)
        #print("err: ",err)
        y_bstr = yhat_fit + err   
        
        
        y_pred_bstr = PP_cv(y_bstr, df_cv)[0]
        
        y_pred_all[i]=y_pred_bstr
        
    #print("bootstrap fininished for t:", t)    
    y_pred = y_pred_all.mean()
    
    yhat_cv_pp[t-T-1]=y_pred
    

rmse_pp=np.sqrt(np.mean((Y_cv_pp-yhat_cv_pp)**2))
print("PP_RMSE_cv: ", rmse_pp)

"""

print("---------------------Model 4: Kalman Filter------------------------")
from KF import kf_train, kf_cv


# crossvalidation
Y_cv_kf =df['Open'][-(N-(T+1)):].values
yhat_cv_kf = np.zeros(N-(T+1))

for t in range(T+1, N):
    # fit
    df_cv = df[t-window:t-1]
    y=df_cv['Open']

    res_fit = kf_train(y)[1]
    yhat_fit = kf_train(y)[0]
    
    
    # bootstrap method: switching residuals
    y_pred_all = np.zeros(num_boot)
    for i in range(0, num_boot):
        err = np.random.choice(res_fit, (window-1,), replace=True)
        y_bstr = yhat_fit + err   
          
        y_pred_bstr = kf_cv(y_bstr) 
        
        y_pred_all[i]=y_pred_bstr
        
    #print("bootstrap fininished for t:", t)    
    y_pred = y_pred_all.mean()
    yhat_cv_kf[t-T-1]=y_pred
    

rmse_kf=np.sqrt(np.mean((Y_cv_kf-yhat_cv_kf)**2))
print("kf_RMSE_cv: ", rmse_kf)



# plot
timevec = np.linspace(1,N-(T+1),N-(T+1))
plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(timevec, Y_cv_ar1, 'b', label = "Y_cv: 50 days")
ax.plot(timevec, yhat_cv_ar1, 'r', label = "yhat_cv ar1")
ax.plot(timevec, yhat_cv_ema, 'g', label = "yhat_cv ema")
# ax.plot(timevec, yhat_cv_pp, 'pink', label = "yhat_cv pp")
ax.plot(timevec, yhat_cv_kf, 'orange', label = "yhat_cv kf")
plt.title('Bootstrap & Cross Validation - Facebook')
ax.legend(loc=2, bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
plt.show()


