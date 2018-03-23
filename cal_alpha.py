# -*- coding: utf-8 -*-

from alphaFuncs import alpha_all
from pathconfig import timeSerialFile, stockListFile, alphaSavePath

def cal_alpha(savepath=alphaSavePath):
    stockList = open(stockListFile).read().split('\n')
    dateList = open(timeSerialFile).read().split('\n')
    dateList = [x for x in dateList if x >= '2017']
    alpha_all(stockList, dateList, savepath=alphaSavePath)
    return 0

if __name__ == "__main__":
    cal_alpha()
