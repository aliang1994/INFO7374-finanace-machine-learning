# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:08:38 2018

MONTE CARLO

@author: Yizhen Zhao
"""
"Data"
import numpy as np
df = 4
X = np.random.standard_t(df, size = 1000)
T = X.shape[0];
X_range = np.sort(X)
print("True value of X mean: mu", np.mean(X))
print("confidence interval of X", X_range[25], X_range[975])

'''
"Data: Log-normal"
mu = 3
se = 1
X = np.random.lognormal(mu, se, 1000)
X_range = np.sort(X)
print("True value of mu:", mu)
print("confidence interval of X", X_range[25], X_range[975])
'''

M = 1000
mu = np.mean(X)
se = np.std(X)
mu_mc = np.zeros(M)
se_mc = np.zeros(M)
t_stat_mc = np.zeros(M)
for i in range(0, M):
    x_mc =  np.random.normal(0,1,T)
    mu_mc[i] = np.mean(x_mc)
    se_mc[i] = np.std(x_mc)/np.sqrt(T)
    t_stat_mc[i] = (mu_mc[i]-mu)/se_mc[i] 
mu_mc = np.sort(mu_mc)
se_mc = np.sort(se_mc)/np.sqrt(T)
t_stat_mc = np.sort(t_stat_mc)

print("confidence interval of mu_mc:", mu_mc[25], mu_mc[975])
print("confidence interval of x through Monte Carlo", mu-t_stat_mc[975]*se, mu-t_stat_mc[25]*se)