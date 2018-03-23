# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd


alpha_neutral = os.listdir(r'G:\short_period_mf\netual_process')
for an in alpha_neutral:
    alpha_ne = pd.read_pickle(r'G:\short_period_mf\netual_process\%s'%an)
    alpha_ne=alpha_ne.melt(id_vars='code')
    alpha_ne=alpha_ne.pivot(index='variable', columns='code', values='value')
    res = alpha_ne.groupby(lambda alpha_ne: alpha_ne[:13]).mean()  
    
 
    