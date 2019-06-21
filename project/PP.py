#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from fbprophet import Prophet
import logging


logging.getLogger().setLevel(logging.CRITICAL)
logging.propagate = False


"""
We did not use this model in Random Forest because running time is too long.
We do have code that's commented out and can run this model.
If you want to run it, install Prophet and pystan first before you run. 
"""

def PP_train(date, Y_train): 
    fb = np.stack((date, Y_train), axis = 1)
    df_fb = pd.DataFrame(fb, columns=['ds','y'])
    model = Prophet(interval_width= 0.70)
    model.fit(df_fb)  
    
    yhat = model.predict()['yhat']
    future_dates = model.make_future_dataframe(periods=1, freq='MS')
    pred = model.predict(future_dates)
    y_pred = pred['yhat'][-1:]
    
    
    """
    rmse=np.sqrt(np.mean(np.power((np.array(Y)-np.array(yhat)),2)))
    r2 = r2_score(df_fb['y'], yhat)   
    print("R Squared:", r2)
    print("RMS:", rmse)    
    model.plot(pred,uncertainty=True)
    """
    
    return yhat, y_pred



