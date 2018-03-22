# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:45:18 2018

@author: wuwangchuxin
"""

import pandas as pd
import pickle
import os

# an = 'netual_alpha_001.pickle'
# alpha_ne.head()

alpha_neutral = os.listdir(r'G:\short_period_mf\netual_process')
for an in alpha_neutral:
    alpha_ne = pd.read_pickle(r'G:\short_period_mf\netual_process\%s'%an)
    alpha_ne=alpha_ne.melt(id_vars='code')
    alpha_ne=alpha_ne.pivot(index='variable', columns='code', values='value')
    res = alpha_ne.groupby(lambda alpha_ne: alpha_ne[:13]).mean()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    