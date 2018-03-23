# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathconfig import industryFile, alphaRawPath, alphaFinalPath

industry = pd.read_csv(industryFile, index_col=0).drop_duplicates()

def resid(x, y):
    return sm.OLS(x, y).fit().resid

def standard(df):
    df = df.apply(lambda x: (x - x.mean()) / x.std())
    df = df.fillna(0)
    df = df.clip(-3, 3)
    return df
    
def neutral(df):
    num_mint = df.shape[1]
    num_inds = industry.shape[1]
    data = pd.concat([df, industry], axis=1).dropna()
    X = data.iloc[:, :num_mint]
    y = data.iloc[:, num_mint:num_mint+num_inds]
    X = X.apply(lambda x:resid(x, y))
    return X

def preProcess():
    files = os.listdir(alphaRawPath)
    for f in files:
        df = pd.read_csv(alphaRawPath + f, index_col=0)
        df = neutral(standard(df))
        df.to_csv(alphaFinalPath + f)
    return 0

if __name__ == "__main__":
    preProcess()
