# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathconfig import alphaRawPath, alphaFinalPath

# TODO 行业和收益率文件
industry = pd.read_pickle('C:/Users/wuwangchuxin/Desktop/yinhua_\
    min/data/industry.pkl').drop_duplicates()

return_data = pd.read_pickle('G:short_period_mf/dailyreturn.pick\
    le').rename(columns={'symbol':'code'})
####

def resid(x, y):
    return sm.OLS(x, y).fit().resid

def standard(df):
    df = df.apply(lambda x: (x - x.mean) / x.std())
    df = df.clip(-3, 3)
    return df
    
def neutral(df):
    num_mint = df.shape[1]
    num_inds = industry.shape[1]
    data = pd.merge(df, industry, on=['code']).dropna()
    X = data.iloc[:, 1:num_mint]
    y = data.iloc[:, num_mint:num_mint+num_inds]
    X = X.apply(lambda x:resid(x, y))
    X['code'] = alpha_data['code']
    return X

def preProcess():
    files = os.listdir(alphaRawPath)
    for f in files:
        df = pd.read_csv(alphaRawPath + f, index_col=0)
        df = neutral(standard(df))
        df.to_csv(alphaFinalPath)
    return 0

if __name__ == "__main__":
    preProcess()
