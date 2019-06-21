# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:07:20 2019
OLS REGRESSION
@author: yizhen zhao
"""
import numpy as np
import scipy.stats as ss

'Choose a sample size'
T = 750
'Generate Sample for Y, X'
mu = (0, 0, 0)
cov = [[1, 0, 0],\
       [0, 1, 0],\
       [0, 0,1]]
F = np.random.multivariate_normal(mu,cov,T)

'Generate Sample for Y, X'
X = np.column_stack([np.ones((T,1)), F])
'N = (750, 4), N[1] = 4'
N = X.shape

'Define Coefficient Matrix, define rank to be N[1] x 1'
beta = np.array([0.56, 2.53, 2.05, 1.78])
beta.shape = (N[1],1)
'Generate a sample of Y'
Y = X@beta+np.random.normal(0,1,(T,1))

'OLS REGRESSION STARTS'
'Linear Regression of Y: T x 1 on'
'Regressors X: T x N'
invXX = np.linalg.inv(X.transpose()@X)
'OLS estimates for coefficients: X x 1'
beta_hat = invXX@X.transpose()@Y
'Predictive value of Y using OLS'
y_hat = X@beta_hat
'Residuals from OLS'
residuals = Y - y_hat
'Variance of residuals'
sigma2 = (1/T)*residuals.transpose()@residuals
'standard deviation of Y or residuals'
sigma = np.sqrt(sigma2)

'variance-covariance matrix of beta_hat'
varcov_beta_hat = (sigma2)*invXX
std_beta_hat = np.sqrt(T*np.diag(varcov_beta_hat))

'Calculate R-square'
R_square = 1- (residuals.transpose()@residuals)/(T*np.var(Y))
adj_R_square = 1-(1-R_square)*(T-1)/(T-N[1])

'Test Each Coefficient: beta_i'
'Null Hypothesis: beta_i = 0'
t_stat = (beta_hat.transpose()-0)/std_beta_hat
p_val_t = 1-ss.norm.cdf(t_stat)

'Test of Joint Significance of Model'
F_stat = (beta_hat.transpose()@np.linalg.inv(varcov_beta_hat)@beta_hat/N[1])/\
         (residuals.transpose()@residuals/(T-N[1]))

p_val_F = 1-ss.f.cdf(F_stat,N[1]-1,T-N[1])