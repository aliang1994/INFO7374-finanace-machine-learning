#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:10:01 2019

@author: aliceliang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_forest import random_forest
import scipy.stats as ss

Y, Y_rf, Y_close = random_forest()
T=Y_rf.shape[0]


"trading rule 1: day trade"
signal_rule1 =  np.zeros(T)
for t in range(0, T):
    if Y_rf[t] > Y[t]:
        signal_rule1[t] = 1  # long signal
    elif Y_rf[t] < Y[t]:
        signal_rule1[t] = -1  # short signal
        
pos_rule1 = signal_rule1 # open and close position every day

"trading rule 2: long short"
signal_rule2 = np.zeros(T)
pos_rule2 = np.zeros(T)
for t in range(0, T):
    if Y_rf[t] > Y[t]:
        signal_rule2[t] = 1  
    elif Y_rf[t] < Y[t]:
        signal_rule2[t] = -1  

for t in range(0, T):
    if t==0:
        pos_rule2[t] = signal_rule2[t]
    elif signal_rule2[t] != signal_rule2[t-1]: 
        # take the first long/short signal as position
        pos_rule2[t] = signal_rule2[t] 
  
"trading rule 3: buy hold"
signal_rule3 =  np.zeros(T)
pos_rule3 = np.zeros(T)
for t in range(0, T):
    if t==0:
        signal_rule3[t] = 1  
    else:
        signal_rule3[t] =  -1
      
pos_rule3 = signal_rule3


"account balances"

init_bal = 1000000 # initial account balance
num_shares = 500 # number of shares being traded in one position


balance_rule1= np.zeros(T)
balance_rule2= np.zeros(T)
balance_rule3= np.zeros(T)

temp_bal1 = init_bal
temp_bal2 = init_bal
temp_bal3 = init_bal
for t in range(0,T):  
    # day trade
    balance_rule1[t] = temp_bal1 - pos_rule1[t]*num_shares*Y[t] + pos_rule1[t]*num_shares*Y_close[t]
    temp_bal1 = balance_rule1[t]
    
    # long short
    balance_rule2[t] = temp_bal2 - pos_rule2[t]*num_shares*Y[t] + pos_rule2[t]*num_shares*Y_close[t]
    temp_bal2 = balance_rule2[t]  
    
    # buy hold
    if t==0:
        # initially hold long position and not close it
        balance_rule3[t] = temp_bal3 - pos_rule3[t]*num_shares*Y[t]  
        temp_bal3 = balance_rule3[t]
    else:
        # closing initial position at time t, aka holdig it for t-1 days
        balance_rule3[t] = temp_bal3 + pos_rule3[0]*num_shares*Y_close[t] 

    
    
# plot account balance
timevec = np.linspace(1,T-1,T-1)
plt.figure(figsize=(30,20))

ax1 = plt.subplot(211)
ax1.plot(timevec, balance_rule1[1:], 'blue', label = "Day Trade")
ax1.plot(timevec, balance_rule2[1:], 'red', label = "Long Short")
ax1.plot(timevec, balance_rule3[1:], 'green', label = "Buy Hold")
ax1.legend(loc=2, bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
plt.title('Trading Strategy Performance - Facebook')
plt.show()



# save to dataframe
combined = np.stack([Y, Y_rf, signal_rule1, pos_rule1, balance_rule1, 
                     signal_rule2, pos_rule2, balance_rule2,   
                     signal_rule3, pos_rule3, balance_rule3], axis = 1)
df = pd.DataFrame(combined, index=np.arange(0,T), 
                  columns=['Y','Y_RF','sig_1','pos_1','bal_1',
                           'sig_2','pos_2','bal_2',
                           'sig_3','pos_3','bal_3'])
df.to_csv('trades.csv')




"analytics: profit and loss, ratios, etc"

# daily profit and loss
balance_rule1 = np.insert(balance_rule1, 0, init_bal)
profit_loss1 =  np.diff(balance_rule1)

balance_rule2 = np.insert(balance_rule2, 0, init_bal)
profit_loss2 =  np.diff(balance_rule2)

balance_rule3 = np.insert(balance_rule3, 0, init_bal)
profit_loss3 =  balance_rule3 - init_bal
           

total_profit1 = sum(x for x in profit_loss1 if x>0)    
total_loss1 = abs(sum(x for x in profit_loss1 if x<0))
cnt_profit1 = sum(1 for x in profit_loss1 if x>0)
cnt_loss1 = sum(1 for x in profit_loss1 if x<0)
        

total_profit2 = sum(x for x in profit_loss2 if x>0)    
total_loss2 = abs(sum(x for x in profit_loss2 if x<0))
cnt_profit2 = sum(1 for x in profit_loss2 if x>0)
cnt_loss2 = sum(1 for x in profit_loss2 if x>0)


if profit_loss3[-1]>0:    
    total_profit3 = profit_loss3[T-1] 
    total_loss3 = 0
else:   
    total_loss3 = profit_loss3[T-1]
    total_profit3 = 0
cnt_profit3 = sum(1 for x in profit_loss3 if x>0)
cnt_loss3 = sum(1 for x in profit_loss3 if x<0)


# daily return in percentage
return1 = (balance_rule1 - init_bal)/init_bal*100
return2 = (balance_rule2 - init_bal)/init_bal*100
return3 = (balance_rule3 - init_bal)/init_bal*100


# ratios, assuming 3 month T bill rate around 2%
sharpe1 = (np.mean(return1) - 2)/np.std(return1-2)
sharpe2 = (np.mean(return2) - 2)/np.std(return2-2)
sharpe3 = (np.mean(return3) - 2)/np.std(return3-2)
sortino1 =(np.mean(return1) - 2)/np.std([x for x in return1 if x<0])
sortino2 =(np.mean(return2) - 2)/np.std([x for x in return2 if x<0])
sortino3 =(np.mean(return3) - 2)/np.std([x for x in return3 if x<0])



print("---------------------------1.DAY TRADE---------------------------")
print("number of trading days: ",T)      
print("total profit: ", total_profit1, "\t total loss: ", total_loss1, "\t profit%: ", total_profit1/init_bal)
print("net profit: ", total_profit1-total_loss1, "\t profit factor: ", total_profit1/total_loss1)
print("profit days: ", cnt_profit1, "\t loss days: ", cnt_loss1, "\t winning rate:", cnt_profit1/T)
print("aver net profit per trade: ", (total_profit1-total_loss1)/T)
print("aver daily return: ", np.mean(return1))
print("daily return std: ", np.std(return1))
print("daily return skewness: ", ss.skew(return1))
print("daily return kurtosis: ", ss.kurtosis(return1))
print("sharpe ratio: ", sharpe1, "\t sortino ratio: ",  sortino1)

den = ss.gaussian_kde(return1) 
xs = np.linspace(-3,3,400)
plt.plot(xs,den(xs))
plt.show()



print("---------------------------2.LONG SHORT---------------------------")
print("number of trading days: ",T)      
print("total profit: ", total_profit2, "\t total loss:", total_loss2, "\t profit%: ", total_profit2/init_bal)
print("net profit", total_profit2-total_loss2, "\t profit factor", total_profit2/total_loss2)
print("profit days: ", cnt_profit2, "\t loss days: ", cnt_loss2, "\t winning rate:", cnt_profit2/T)
print("aver net profit per trade: ", (total_profit2-total_loss2)/T)
print("aver daily return: ", np.mean(return2))
print("daily return std: ", np.std(return2))
print("daily return skewness: ", ss.skew(return2))
print("daily return kurtosis: ", ss.kurtosis(return2))
print("sharpe ratio: ", sharpe2, "\t sortino ratio: ",  sortino2)

den = ss.gaussian_kde(return2) 
xs = np.linspace(-3,3,400)
plt.plot(xs,den(xs))
plt.show()



print("---------------------------3.BUY HOLD---------------------------")
print("number of trading days: ",T)      
print("total profit: ", total_profit3, "\t total loss:", total_loss3, "\t profit%: ", total_profit3/init_bal)
print("net profit", total_profit3-total_loss3, "\t profit factor", total_profit3/total_loss3)
print("profit days: ", cnt_profit3, "\t loss days: ", cnt_loss3, "\t winning rate:", cnt_profit3/T)
print("aver net profit per trade: ", (total_profit3-total_loss3)/T)
print("aver daily return: ", np.mean(return3))
print("daily return std: ", np.std(return3))
print("daily return skewness: ", ss.skew(return3))
print("daily return kurtosis: ", ss.kurtosis(return3))
print("sharpe ratio: ", sharpe3, "\t sortino ratio: ",  sortino3, "--no negative return under buy hold except for initial long position")

den = ss.gaussian_kde(return3) 
xs = np.linspace(-1,6,400)
plt.plot(xs,den(xs))
plt.show()
