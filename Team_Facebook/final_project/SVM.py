#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:47:44 2019

@author: aliceliang
"""

from sklearn.svm import SVR  # package that we used


def svm_train(X_train, Y_train, X_pred):  
    """
    # This below model has better performance (RMSE as low as AR1), but running time is also very much longer
    
    svr_model = SVR(kernel='poly', C=1, gamma=10, degree=1, epsilon=.1, coef0=1)
    """
    
    
    "Acceptably higher RMSE (< 6) but much less running time:"
    
    svr_model = SVR(kernel='rbf', gamma=0.0005)
    result = svr_model.fit(X_train, Y_train)
 
    y_hat = result.predict(X_train)  
    y_pred = result.predict(X_pred)
    
    return y_hat, y_pred