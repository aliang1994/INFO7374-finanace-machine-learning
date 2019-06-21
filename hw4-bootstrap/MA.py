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

"""
sns.set(style='darkgrid', context='talk', palette='Dark2')
result = pd.read_excel(io='~/Desktop/FB.xlsx', sheet_name='Sheet1', header=0, 
                   names=['Date','Open','High','Low','Close','Volume','Return'], 
                   index_col=None)

result = pd.read_excel(io='~/Desktop/7374/FB.xlsx', sheet_name='Sheet1', header=0, 
                  names=['Date','Open','High','Low','Close','Volume'], index_col=None)

train =result[:500]
test = result[500:]
"""

def EMA_train(train):

    # Indexing the date to plot a good time-series on the graph
    #train['Date'] = pd.to_datetime(train['Date'])
    #train.index = train['Date']
    
    # Calculating the short-window simple moving average. Window = 20 Days
    #short_rolling = train['Close'].rolling(window=5).mean()
    
    # Calculating the long-window simple moving average. Window: 100 Days
    #long_rolling = train['Close'].rolling(window=100).mean()
    
    """
    #Plot rolling statistics:
    fig, ax = plt.subplots(figsize=(17,10))
    orig= ax.plot(train['Close'],color='blue',label='Original')
    short_mean=ax.plot(short_rolling,color='orange',label='Short Rolling Mean')
    long_mean=ax.plot(long_rolling,color='green',label='Long Rolling Mean')
    #plt.legend(loc='best')
    ax.legend(loc='best')
    ax.set_title('Simple Moving Average for 20 and 100 Days')
    ax.set_xlabel('Obsevation Date')
    ax.set_ylabel('Price in $')
    """
    # Calculating a 20-days span EMA. adjust=False specifies that we 
    # are interested in the recursive calculation mode.
    
    ema_short = train['Close'].ewm(span=5, adjust=False).mean()
    """
    fig, ax = plt.subplots(figsize=(17,10))
    orig= ax.plot(train['Close'],color='blue',label='Actual Price')
    short_mean_SMA=ax.plot(short_rolling,color='red',label='Short Rolling Mean SMA')
    short_mean_EMA=ax.plot(ema_short,color='yellow',label='Short Rolling Mean EMA')
    
    
    ax.legend(loc='best')
    ax.set_title('EWMA - 20 Days Span')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price in $')
    """
    # OLS Method implementation yt=beta2*EMA(20)+uhatt
    
    T=train.shape[0]
    
    # regression variables
    Y = train['Close'].values
        
    X = ema_short
    X = np.column_stack([np.ones((T,1)),X])
        
    N = X.shape[1]
        
        
    invXTX = np.linalg.inv(X.transpose()@X)
    
     # ols estimates
     # beta = inverse(X'X)(X'Y)
    beta_hat = invXTX@X.transpose()@Y
    #print("EMWA betahat: ", beta_hat)
    
     # predictive value of y using OLS
    y_hat = X@beta_hat
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
    return beta_hat, res, y_hat, ema_short;

def EMA_cv(y_bstr, train):
    # Calculating a 20-days span EMA. adjust=False specifies that we 
    # are interested in the recursive calculation mode.
    
    ema_short = train['Close'].ewm(span=5, adjust=False).mean()
      
    # OLS Method implementation yt=beta2*EMA(20)+uhat
    T=train.shape[0]
    
    # regression variables
    Y = y_bstr      
    X = ema_short
    X = np.column_stack([np.ones((T,1)),X])     
    N = X.shape[1]
        
        
    invXTX = np.linalg.inv(X.transpose()@X)
    beta_hat = invXTX@X.transpose()@Y
    #print("EMWA betahat: ", beta_hat)
    
    # predictive value of y using OLS
    y_hat = X@beta_hat
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
    #t_stat = (beta_hat.transpose()-0)/std_cov_beta
    #p_val_t = 1-ss.norm.cdf(t_stat)
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
    return beta_hat, res, y_hat, ema_short;
    
    
    return beta_hat, res, y_hat

#res = EMA_train(result)
