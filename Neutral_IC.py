# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

industry = pd.read_pickle\
    ('C:/Users/wuwangchuxin/Desktop/yinhua_min/data/industry.pkl').drop_duplicates()

return_data = pd.read_pickle\
    ('G:short_period_mf/dailyreturn.pickle').rename(columns={'symbol':'code'})

def resid(x, y):
    return sm.OLS(x, y).fit().resid
    
def Neutral_process(alpha_data, saf):
    num_mint = alpha_data.shape[1]
    num_inds = industry.shape[1]
    data = pd.merge(alpha_data, industry, on=['code']).dropna()
    X = data.iloc[:, 1:num_mint]
    y = data.iloc[:, num_mint:num_mint+num_inds]
    X = X.apply(lambda x:resid(x, y))
    X['code'] = alpha_data['code']
    output = open(r'G:/short_period_mf/netual_process/netual_%s'%saf[-16:],'wb')
    pickle.dump(X,output)
    output.close()
    return X

#def IC_computing(alpha_data, saf):
#    data = pd.merge(alpha_data,return_data,on = ['code'])
#    dailyReturn = data.daily_return
#    factors = Neutral_process(alpha_data,saf)
#    IC = dailyReturn.corr(factors, method='spearman')
#    output = open(r'G:/short_period_mf/ic_value/ic_%s'%saf[-16:],'wb')
#    pickle.dump(IC,output)
#    output.close()    
#    return 0


################
standard_alpha = os.listdir(r'G:/short_period_mf/alpha_min_stand')
#for saf in standard_alpha:
#    alpha_d = pd.read_pickle(r'G:/short_period_mf/alpha_min_stand/%s'%saf)
#    IC_computing(alpha_d, saf)

for saf in standard_alpha:
    alpha_d = pd.read_pickle(r'G:/short_period_mf/alpha_min_stand/%s'%saf)
    Neutral_process(alpha_d,saf)