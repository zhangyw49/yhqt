# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:42:21 2018

@author: wuwangchuxin
"""
#import time
import numpy as np
import pandas as pd
import pickle
import os
import statsmodels.api as sm

#os.chdir(r'E:\alpha_min')

#start=time.clock()
def standard_progress():
    filenameList = os.listdir(r'E:\alpha_min')
#    filenameList = ['alpha_001.csv']
    for filename in filenameList:
        #首先对因子值进行空值删除和标准化处理
#        start=time.clock()
        data = pd.read_csv(filename)
        data = pd.concat([data.iloc[:,0],data.iloc[:,7:]],axis=1)
        data_c = data.fillna(0)
        data_b = data_c.iloc[:,1:]
        data_d = np.array(data_b)
        x_mean = data_d.mean(axis = 0)
        x_std = data_d.std(axis = 0)
        for i in range(len(data_c.columns)-1):
            for j in range(len(data_c)-1):
                if data_d[j][i] > (x_mean[i]+1.65*x_std[i]):
                    data_d[j][i] = (x_mean[i]+1.65*x_std[i])
                elif data_d[j][i] < (x_mean[i]+1.65*x_std[i]):
                    data_d[j][i] = (x_mean[i]-1.65*x_std[i])
        data_d=pd.DataFrame(data_d,index=list(data_c['code']),columns=list(data_c.columns)[1:])
        data_d.reset_index(inplace=True)
        data_d = data_d.rename(columns={'index':'code'})
        # 存到移动硬盘里
        output = open(r'G:\short_period_mf\alpha_min_stand\standard_%s.pickle'%filename[:9],'wb')
        pickle.dump(data_d,output)
        output.close()
#        end = time.clock()
    return None

#def Netual_process(alpha_data,industry,saf):
#    new_data = pd.merge(alpha_data,industry,on = ['code'])
#    new_data2 = pd.DataFrame(new_data.code)
#    for i in range(len(new_data.columns.tolist())-1):
#        y = new_data.iloc[:,i+1].as_matrix()
#        X = new_data.iloc[:,-30:].as_matrix()
#        model = sm.OLS(X,y)
#        results = model.fit()
#        Betas = results.params #这里没有常数项，所有参数均为回归系数
#        new_factors = pd.DataFrame(y - X.dot(Betas.T))
#        new_data2 = pd.concat([new_data2,new_factors],axis = 1)
#    output = open(r'G:\short_period_mf\netual_process\netual_%s'%saf[-16:],'wb')
#    pickle.dump(new_data2,output)
#    output.close()
#    return None
#
#def IC_computing(path,industry):
#    data = pd.read_pickle(path)
#    factors = Netual_process(data,industry)
#    IC = minuteReturn.corr(factors)
#    output = open(r'G:\short_period_mf\ic_value\ic_%s'%saf[-16:],'wb')
#    pickle.dump(IC,output)
#    output.close()    
#    return None

industry2=pd.read_pickle(r'C:\Users\wuwangchuxin\Desktop\yinhua_min\data\industry.pkl')
industry3 = industry2.drop_duplicates()
standard_alpha = os.listdir(r'G:\short_period_mf\alpha_min_stand')
for saf in standard_alpha:
    path = r'G:\short_period_mf\alpha_min_stand\%s'%saf
    IC_computing(path,industry)


for saf in standard_alpha:
    alpha_d = pd.read_pickle(r'G:\short_period_mf\alpha_min_stand\%s'%saf)
    Netual_process(alpha_d,industry3,saf)
#
#paths = os.listdir(r'G:\short_period_mf\netual_process')
#for pat in paths:
#    IC_computing(pat,industry)


industry3.to_csv(r'C:\Users\wuwangchuxin\Desktop\industry3.csv')

alpha_d2.to_csv(r'C:\Users\wuwangchuxin\Desktop\netual_alpha001_part.csv')
alpha_d2 = alpha_d[:2]





#
#def IC_computing():
#    data = pd.read_pickle(r'G:\alpha_min_stand\standard_Alpha_001.pickle')
#    return_data = pd.read_pickle('dailyreturn.pickle')
#    return_data = return_data.reset_index()
#    new_data = pd.merge(data,return_data,on = ['code'])
#    dailyReturn = new_data.daily_return
#    factor = new_data.ix[:,0:-1].mean(1)
##    print(factor)
#    IC = dailyReturn.corr(factor)
##    print(IC)
#    return None
#
#
#def dfEWREGBETA(df1, sr, n, halflife=60):
#	df1 = df1.ewm(halflife).mean()
#	sr2 = sr.ewm(halflife).mean()
#    condition_1 = isinstance(sr, np.ndarray)
#    condition_2 = isinstance(sr, pd.core.series.Series)
#    temp = df1.copy()
#    if condition_1:
#        temp.rolling(n).apply(lambda y: stats.linregress(x=sr, y=y)[0])
#    if condition_2:
#        for i in range(0, len(df1) - n):
#            temp.iloc[i:i + n, :].apply(lambda y: stats.linregress(x=sr.iloc[i:i + n], y=y)[0])
#    return temp



















