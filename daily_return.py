# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:26:56 2018

@author: wuwangchuxin
"""

import feather as ft
import pandas as pd
import pickle

daily = ft.read_dataframe(r'E:\marketData.feather')

daily.head()

daily_2017 = daily[daily['date']>='2017-01-01']
daily_2017 = daily_2017[['date','symbol','close','preClose']]

daily_2017['daily_return'] = (daily_2017['close']-daily_2017['preClose'])*100/daily_2017['preClose']

output=open(r'C:\Users\wuwangchuxin\Desktop\dailyreturn.pickle','wb')
pickle.dump(daily_2017,output) 
output.close()