#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import r2_score
import logging


logging.getLogger().setLevel(logging.CRITICAL)
logging.propagate = False


def PP_train(train): 
    
    train['date'] = train.index
    df_fb = pd.concat([train['date'], train['Open']], axis = 1)
    df_fb = df_fb.rename(columns={'date': 'ds','Open': 'y'})
    

    model = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.1, interval_width= 0.70)
    model.fit(df_fb)  
    
    yhat = model.predict()['yhat']

    future_dates = model.make_future_dataframe(periods=6, freq='MS')
    pred = model.predict(future_dates)
    
    
    df= df_fb
    df['yhat']=yhat.values
    res = df['y']-df['yhat']
    #print("res here:", res)
    #rms=np.sqrt(np.mean(np.power((np.array(Y)-np.array(yhat)),2)))

    #r2 = r2_score(df_fb['y'], yhat)
    
    #print("R Squared:", r2)
    #print("RMS:", rms)
    
    #model.plot(pred,uncertainty=True)

    return res, yhat


def PP_cv(y_bstr, train):
    logging.getLogger('fbprophet').setLevel(logging.ERROR)
    train['date'] = train.index
    train['y_bstr']=y_bstr.values
    
    df_fb = pd.concat([train['date'], train['y_bstr']], axis = 1)
    df_fb = df_fb.rename(columns={'date':'ds','y_bstr':'y'})

    model = Prophet(interval_width= 0.70)
    model.fit(df_fb)  
    
    Y=df_fb['y']
    yhat = model.predict()['yhat']
    
    res = Y-yhat
    
    future_dates = model.make_future_dataframe(periods=1, freq='MS')
    pred = model.predict(future_dates)
    y_pred = pred['yhat'][-1:]
    #print("pred:",y_pred)


    return y_pred, res


