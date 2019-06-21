
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

import math

import pandas as pd

import numpy as np

from pandas.core.frame import DataFrame

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.svm import SVR

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

merge=pd.read_csv('/Users/gaojunling/Desktop/application of ML in Finance/mergeresult.csv')

merge

def svm_regression_diff(ori_traget):
    ori_traget['Date'] = pd.to_datetime(ori_traget['Date'])
    ori_traget=ori_traget.set_index('Date')
    traindays = ori_traget.iloc[0:(int(0.75*len(ori_traget))),:]
    traindays = traindays.reset_index()
    testdays =  ori_traget.iloc[(int(0.75*len(ori_traget))):(int(len(ori_traget))),:]
    testdays = testdays.reset_index()
    train_FB_daily_return = ori_traget.iloc[0:(int(0.75*len(ori_traget))),10]
    test_FB_daily_return = ori_traget.iloc[(int(0.75*len(ori_traget))):(int(len(ori_traget))),10]
    X_ = np.hstack((traindays.iloc[:,[1,2,3,4]].values,np.expand_dims(np.array(train_FB_daily_return.values),axis =1)))
    "train X: Open High Low Close"
    X = X_[:,[0,1,2,3]]
    X_test_ = np.hstack((testdays.iloc[:,[1,2,3,4]].values,np.expand_dims(np.array(test_FB_daily_return.values),axis =1)))
    "test X: Open High Low Close"
    X_test = X_test_[:,[0,1,2,3]]
    "Y_:daily return rate"
    Y_ = X_[:,3]
    "TRAIN y set:  Day Open Price"
    Yprice_ = X_[:,0]
    Y = np.expand_dims(Y_,axis=1)
    Y_test_ = X_test_[:,3]
    Y_test = np.expand_dims(Y_test_,axis =1)
    Ypricetrain = np.expand_dims(Yprice_,axis=1)
    "test y set:  Day Open Price"
    
    Ypricetest_ = X_test[:,0]
    
    Ypricetest = np.expand_dims(Ypricetest_,axis =1)
    "Difftrain_: difference of High and Low"
    Difftrain_ = X[:,1]-X[:,2]
    Difftrain__ = np.vstack((Difftrain_,X[:,3])).T
    Difftrain = Difftrain__
    Difftest_ = X_test[:,1]-X_test[:,2]
    Difftest__ = np.vstack((Difftest_,X_test[:,3])).T
    Difftest = Difftest__
    svr_lin_diff_price = SVR(kernel='poly', C=1, gamma=10, degree=1, epsilon=.1,
               coef0=1)

    svr_lin_diff_price.fit(Difftrain, Ypricetrain)
    return (svr_lin_diff_price.predict(Difftest))

