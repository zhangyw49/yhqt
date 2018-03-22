# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:49:28 2018

@author: wuwangchuxin
"""


from alphaFuncs import *
import math
import numpy as np
import pandas as pd
from scipy import stats
from functools import reduce
import feather as ft

#test =  ft.read_dataframe(r'G:\1m_data\1\SH600000.feather', nthreads=100)

#minute_file_path = '/home/tober/work/yinhua/minute/'
#timeSerialFile='/home/tober/work/mfm2/date/trade.date'

minute_file_path = 'G:\\1m_data\\1\\'
timeSerialFile=r'C:\Users\wuwangchuxin\Desktop\yinhua_min\data\trade.date'

#stockList=['600004.SH', '600000.SH', '600006.SH', '600007.SH', '600008.SH']
code_HS300 = pd.read_excel(r'C:\Users\wuwangchuxin\Desktop\yinhua_min\data\data_mkt.xlsx',\
              sheetname='HS300')
stockList = list(code_HS300['code'][:])
#dateList=['2015-04-15 09:46:00', '2015-04-15 09:47:00']
#dateList = pd.read_pickle(r'C:\Users\wuwangchuxin\Desktop\yinhua_min\data\min_series.pickle')
#dateList = pd.read_pickle(r'C:\Users\wuwangchuxin\Desktop\yinhua_min\data\trade.date')
dateList = open(r'C:\Users\wuwangchuxin\Desktop\yinhua_min\data\trade.date').read().split('\n')
#####分钟线：从2010年起至

alpha_all(stockList, dateList, savepath='E:\\alpha_min\\')











