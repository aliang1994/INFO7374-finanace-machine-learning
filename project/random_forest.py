#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:14:20 2019

@author: aliceliang
"""
import numpy as np
import matplotlib.pyplot as plt
from bootstrap import ar1_bstr, ema_bstr, kf_bstr, svm_bstr, get_Y
#from bootstrap import pp_bstr

def random_forest():
    yhat_ar1, rmse_ar1 = ar1_bstr()    
    yhat_ema, rmse_ema = ema_bstr()
    #yhat_pp, rmse_pp = pp_bstr()  # too slow 
    yhat_kf, rmse_kf = kf_bstr()
    yhat_svm, rmse_svm = svm_bstr()
    
    
    Y, Y_close = get_Y()
    T=Y.shape[0]
    
    "random forest voting"
    Y_rf = np.zeros(T)
    for t in range(0, T):
        if min(abs(yhat_ar1[t]-Y[t]),abs(yhat_ema[t]-Y[t]),
               abs(yhat_kf[t]-Y[t]))==abs(yhat_ar1[t]-Y[t]):
            Y_rf[t] = yhat_ar1[t]
        elif min(abs(yhat_ar1[t]-Y[t]),abs(yhat_ema[t]-Y[t]),
                 abs(yhat_kf[t]-Y[t]))==abs(yhat_ema[t]-Y[t]):
            Y_rf[t] = yhat_ema[t]
        elif min(abs(yhat_ar1[t]-Y[t]),abs(yhat_ema[t]-Y[t]),
                 abs(yhat_kf[t]-Y[t]))==abs(yhat_kf[t]-Y[t]):
            Y_rf[t] = yhat_kf[t]          
        else:
            Y_rf[t] = yhat_svm[t]  
    rmse_rf=np.sqrt(np.mean((Y-Y_rf)**2))
    
    print("ar1_RMSE: ", rmse_ar1)
    print("ema_RMSE: ", rmse_ema)
    #print("PP_RMSE: ", rmse_pp)
    print("kf_RMSE: ", rmse_kf)
    print("svm_RMSE: ", rmse_svm)
    print("random forest rmse: ", rmse_rf)
    
    
    # plot 
    timevec = np.linspace(1,T,T)
    plt.figure(figsize=(30,20))
    
    ax = plt.subplot(211)
    ax.plot(timevec, Y, 'blue', label = "Y: original")
    ax.plot(timevec, yhat_ar1, 'red', label = "yhat ar1")
    ax.plot(timevec, yhat_ema, 'green', label = "yhat ema")
    #ax.plot(timevec, yhat_pp, 'pink', label = "yhat pp")
    ax.plot(timevec, yhat_svm, 'purple', label = "yhat svm")
    ax.plot(timevec, yhat_kf, 'orange', label = "yhat kf")
    plt.title('Single Model Prediction - Facebook')
    ax.legend(loc=2, bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
    
    
    ax = plt.subplot(212)
    ax.plot(timevec, Y, 'blue', label = "Y: original")
    ax.plot(timevec, Y_rf, 'red', label = "Y_rf")
    plt.title('Random Forest Prediction - Facebook')
    ax.legend(loc=2, bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
    plt.show()
    
    return Y, Y_rf, Y_close