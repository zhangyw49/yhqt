# -*- coding: utf-8 -*-

###############################################################################
import math
import numpy as np
import pandas as pd
from scipy import stats
from functools import reduce
import feather as ft
from pathconfig import *
###############################################################################
###############################################################################

def alpha_all(stockList, dateList, savepath):
    for i in range(1，192):
        try:
            tmp = eval('alpha_{:03}(stockList, dateList)'.format(i))
            tmp.to_csv(savepath + 'alpha_{:03}.csv'.format(i))
            print(i,'Done')
        except Exception as e:
            print(i,e)   

def alpha_001(stockList, dateList):
    # (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    fields = 'open, close, volume'
    offday = -7

    openData, closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(dfDelta(dfLog(volData + 1), 1))
    rank_2 = csRank((closeData - openData) / (openData + 0.001))
    result = -1 * rollCorr(rank_1, rank_2, 6)
    
    return result.T[dateList]


def alpha_002(stockList, dateList):
    # (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    fields = 'close, high, low'
    offday = -1

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)
    
    temp_1 = (closeData - lowData) - (highData - closeData)
    temp_2 = highData - lowData
    result = -1 * dfDelta(temp_1 / (temp_2 + 0.001), 1)

    return result.T[dateList]


def alpha_003(stockList, dateList):
    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    fields = 'close, high, low'
    offday = -7

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    condition_1 = closeData == delay_1
    condition_2 = closeData > delay_1
    temp_1 = dfTripleOperation(condition_1, 0,
                               dfTripleOperation(condition_2, dfSmaller(lowData, delay_1), dfLarger(highData, delay_1)))
    result = dfSum(temp_1, 6)

    return result.T[dateList]


def alpha_004(stockList, dateList):
    # (STD(CLOSE, 8) / (SUM(CLOSE, 8) / 8)) * (STD(VOLUME, 8) / (SUM(VOLUME, 8) / 8)) 
    fields = 'close, volume'
    offday = -10
    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    sum_1 = dfSum(closeData, 8) / 8.0
    std_1 = dfStd(closeData, 8)
    sum_2 = dfSum(volData, 8) / 8.0
    std_2 = dfStd(volData, 8)

    result = (std_1 / sum_1) * (std_2 / sum_2)

    return result.T[dateList]


def alpha_005(stockList, dateList):
    # (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    fields = 'high, volume'
    offday = -13
    highData, volData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(tsRank(volData, 5), tsRank(highData, 5), 5)
    result = -1 * dfMax(corr_1, 3)

    return result.T[dateList]


def alpha_006(stockList, dateList):
    # (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    fields = 'open, high'
    offday = -4
    openData, highData = generateDataFrame(stockList, dateList, fields, offday)

    delta_1 = dfDelta(openData * 0.85 + highData * 0.15, 4)
    result = -1 * csRank(dfSign(delta_1))

    return result.T[dateList]


def alpha_007(stockList, dateList):
    # ((RANK(TSMAX((VWAP - CLOSE), 3)) + RANK(TSMIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    fields = 'close, volume, vwap'
    offday = -3

    closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    temp = vwapData - closeData
    rank_1 = csRank(dfMax(temp, 3))
    rank_2 = csRank(dfMin(temp, 3))
    rank_3 = csRank(dfDelta(volData, 3))
    result = (rank_1 + rank_2) * rank_3

    return result.T[dateList] 


def alpha_008(stockList, dateList):
    # RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    fields = 'high, low, vwap'
    offday = -4

    highData, lowData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (highData + lowData) / 2.0 * 0.2
    temp_2 = vwapData * 0.8
    result = csRank(-1 * dfDelta(temp_1 + temp_2, 4))

    return result.T[dateList]


def alpha_009(stockList, dateList):
    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    fields = 'high, low, volume'
    offday = -8

    highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (highData.diff(1) + lowData.diff(1)) / 2.0
    temp_2 = (highData - lowData) / (volData + 1)
    result = tsSma(temp_1 * temp_2, 7, 2)

    return result.T[dateList]


def alpha_010(stockList, dateList):
    # (RANK(TSMAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    fields = 'close, p_change'
    offday = -25

    closeData, retData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = retData < 0
    std_1 = dfStd(retData, 20)
    result = csRank(dfMax(dfTripleOperation(condition_1, std_1, closeData) ** 2, 5))

    return result.T[dateList]


def alpha_011(stockList, dateList):
    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)

    fields = 'close, high, low, volume'
    offday = -6

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (closeData - lowData) - (highData - closeData)
    temp_2 = (highData - lowData + 0.001) * (volData + 1)
    result = dfSum(temp_1 / temp_2, 6)

    return result.T[dateList]


def alpha_012(stockList, dateList):
    # (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    fields = 'open, close, vwap'
    offday = -10

    openData, closeData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(openData - dfSum(vwapData, 10) / 10.0)
    rank_2 = csRank(dfABS(closeData - vwapData))
    result = -1 * rank_1 * rank_2

    return result.T[dateList]


def alpha_013(stockList, dateList):
    # (((HIGH * LOW)^0.5) - VWAP)
    fields = 'high, low, vwap'
    offday = 0

    highData, lowData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    result = (highData * lowData + 0.01) ** 0.5 - vwapData

    return result.T[dateList]


def alpha_014(stockList, dateList):
    # CLOSE - DELAY(CLOSE,5)
    fields = 'close'
    offday = -5
    
    closeData = generateDataFrame(stockList, dateList, fields, offday)
    
    result = dfDelta(closeData, 5)

    return result.T[dateList]


def alpha_015(stockList, dateList):
    # OPEN/DELAY(CLOSE,1)-1
    fields = 'open, close'
    offday = -1

    openData, closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = openData / (dfDelay(closeData, 1) + 0.001) - 1

    return result.T[dateList]


def alpha_016(stockList, dateList):
    # (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    fields = 'volume, vwap'
    offday = -10

    volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(rollCorr(csRank(volData), csRank(vwapData), 5))
    result = -1 * dfMax(rank_1, 5)

    return result.T[dateList]


def alpha_017(stockList, dateList):
    # RANK((VWAP - TSMAX(VWAP, 15))) * DELTA(CLOSE, 5)
    fields = 'close, vwap'
    offday = -15

    closeData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(vwapData - dfMax(vwapData, 15))
    delta_1 = dfDelta(closeData, 5)
    result = rank_1 * delta_1

    return result.T[dateList]


def alpha_018(stockList, dateList):
    # CLOSE/DELAY(CLOSE,5)
    fields = 'close'
    offday = -5

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = closeData / (dfDelay(closeData, 5) + 0.001)

    return result.T[dateList]


def alpha_019(stockList, dateList):
    # (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
    fields = 'close'
    offday = -5

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 5)
    condition_1 = closeData < delay_1
    condition_2 = closeData == delay_1
    result = dfTripleOperation(condition_1, (closeData - delay_1) / (delay_1 + 0.001),
                               dfTripleOperation(condition_2, 0, (closeData - delay_1) / (closeData + 0.001)))

    return result.T[dateList]


def alpha_020(stockList, dateList):
    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    fields = 'close'
    offday = -6

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(closeData, 6) / (dfDelay(closeData, 6) + 0.001) * 100

    return result.T[dateList]


def alpha_021(stockList, dateList):
    # REGBETA(MEAN(CLOSE,6),SEQUENCE(6),6)
    fields = 'close'
    offday = -12

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfREGBETA(dfMean(closeData, 6), np.arange(1, 7), 6)

    return result.T[dateList]


def alpha_022(stockList, dateList):
    # SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    fields = 'close'
    offday = -19

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    mean_1 = dfMean(closeData, 6) + 0.001
    temp_1 = (closeData - mean_1) / mean_1
    result = tsSma(temp_1 - dfDelay(temp_1, 3), 12, 1)

    return result.T[dateList]


def alpha_023(stockList, dateList):
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?
    # STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    std_1 = dfStd(closeData, 20)
    condition_1 = closeData > delay_1
    condition_2 = closeData <= delay_1

    sma_1 = tsSma(dfTripleOperation(condition_1, std_1, 0), 20, 1)
    sma_2 = tsSma(dfTripleOperation(condition_2, std_1, 0), 20, 1)
    result = sma_1 / (sma_1 + sma_2 + 0.001) * 100

    return result.T[dateList]


def alpha_024(stockList, dateList):
    # SMA(CLOSE-DELAY(CLOSE,5),5,1)
    fields = 'close'
    offday = -10

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = tsSma(dfDelta(closeData, 5), 5, 1)

    return result.T[dateList]


def alpha_025(stockList, dateList):
    # ((-1 * RANK((DELTA(CLOSE, 7) *
    # (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) *
    # (1 + RANK(SUM(RET, 210))))
    fields = 'close, volume, p_change'
    offday = -240

    closeData, volData, retData = generateDataFrame(stockList, dateList, fields, offday)

    decay_1 = decayLinear(volData /(dfMean(volData, 20) + 1), 9)
    temp_1 = -1 * csRank(dfDelta(closeData, 7) * (1 - csRank(decay_1)))
    result = temp_1 * (1 + csRank(dfSum(retData, 210)))

    return result.T[dateList]


def alpha_026(stockList, dateList):
    # ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
    fields = 'close, vwap'
    offday = -235

    closeData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = dfSum(closeData, 7) / 7.0 - closeData
    temp_2 = rollCorr(vwapData, dfDelay(closeData, 5), 230)
    result = temp_1 + temp_2

    return result.T[dateList] 


def alpha_027(stockList, dateList):
    # WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+
    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    fields = 'close'
    offday = -106

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 3) + 0.001
    delay_2 = dfDelay(closeData, 6) + 0.001
    temp_1 = (closeData - delay_1) / delay_1 * 100 + (closeData - delay_2) / delay_2 * 100
    result = tsWma(temp_1, 12)

    return result.T[dateList]


def alpha_028(stockList, dateList):
    # 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-
    # 2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    fields = 'close, high,  low'
    offday = -30

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = closeData - dfMin(lowData, 9)
    temp_2 = dfMax(highData, 9) - dfMin(lowData, 9) + 0.001
    sma_1 = tsSma(temp_1 / temp_2 * 100, 3, 1)
    sma_2 = tsSma(sma_1, 3, 1)
    result = 3 * sma_1 - 2 * sma_2

    return result.T[dateList]


def alpha_029(stockList, dateList):
    # (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    fields = 'close, volume'
    offday = -6

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(closeData, 6) / (dfDelay(closeData, 6) + 0.001) * volData

    return result.T[dateList]


def alpha_030(stockList, dateList):
    # WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,Volume,60))^2,20)
    fields = 'close, volume'
    offday = -81

    closeData, volumeData = generateDataFrame(stockList, dateList, fields, offday)
    temp = dfREGRESI(closeData.pct_change(), volumeData.mean(axis=1) + 1, 60)

    result = tsWma(temp ** 2, 20)

    return result.T[dateList]


def alpha_031(stockList, dateList):
    # (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    fields = 'close'
    offday = -12

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = (closeData - dfMean(closeData, 12)) / (dfMean(closeData, 12) + 0.001) * 100

    return result.T[dateList]


def alpha_032(stockList, dateList):
    # (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    fields = 'high, volume'
    offday = -6

    highData, volData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(csRank(highData), csRank(volData), 3)
    result = -1 * dfSum(csRank(corr_1), 3)

    return result.T[dateList]


def alpha_033(stockList, dateList):
    # ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) *
    # RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *
    # TSRANK(VOLUME, 5))
    fields = 'low, volume, p_change'
    offday = -240

    lowData, volData, retData = generateDataFrame(stockList, dateList, fields, offday)

    min_1 = dfMin(lowData, 5)
    temp_1 = -1 * min_1 + dfDelay(min_1, 5)
    rank_1 = csRank((dfSum(retData, 240) - dfSum(retData, 20)) / 220.0)
    result = temp_1 * rank_1 * tsRank(volData, 5)

    return result.T[dateList]


def alpha_034(stockList, dateList):
    # MEAN(CLOSE,12)/CLOSE
    fields = 'close'
    offday = -12

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfMean(closeData, 12) / closeData

    return result.T[dateList]


def alpha_035(stockList, dateList):
    # (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +
    # (OPEN *0.35)), 17),7))) * -1)
    fields = 'open, close, volume'
    offday = -24

    openData, closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear(dfDelta(openData, 1), 15))
    corr_1 = rollCorr(volData, openData * 0.65 + closeData * 0.35, 17)
    rank_2 = csRank(decayLinear(corr_1, 7))
    result = -1 * dfSmaller(rank_1, rank_2)

    return result.T[dateList]


def alpha_036(stockList, dateList):
    # RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)
    fields = 'volume, vwap'
    offday = -8

    volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(csRank(volData), csRank(vwapData), 6)
    result = csRank(dfSum(corr_1, 2))

    return result.T[dateList]


def alpha_037(stockList, dateList):
    # (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    fields = 'open, p_change'
    offday = -15

    openData, retData = generateDataFrame(stockList, dateList, fields, offday)

    sum_1 = dfSum(openData, 5)
    sum_2 = dfSum(retData, 5)
    result = -1 * dfDelta(sum_1 * sum_2, 10)

    return result.T[dateList]


def alpha_038(stockList, dateList):
    # (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)): DELTA(HIGH, 2))
    fields = 'high'
    offday = -20

    highData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = dfSum(highData, 20) / 20.0 < highData
    delta = dfDelta(highData, 2)
    result = dfTripleOperation(condition_1, -delta, delta)
    result = result.applymap(lambda x: 0 if x is False else x)

    return result.T[dateList]


def alpha_039(stockList, dateList):
    # ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),
    # SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
    fields = 'open, close, volume, vwap'
    offday = -243

    openData, closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    vwap_openData = vwapData * 0.3 + openData * 0.7
    volumeMean = dfMean(volData, 180)
    volumeMeanSum = dfSum(volumeMean, 37)
    corrData = rollCorr(vwap_openData, volumeMeanSum, 14)
    decayData = decayLinear(corrData, 12)
    decayRank = csRank(decayData)
    closeDelta = dfDelta(closeData, 2)
    closeDecay = decayLinear(closeDelta, 8)
    closeRank = csRank(closeDecay)
    result = -1 * (closeRank - decayRank)

    return result.T[dateList][dateList]


def alpha_040(stockList, dateList):
    # alpha_040 = SUM((CLOSE > DELAY(CLOSE, 1)?VOLUME:0), 26) / SUM((CLOSE <= DELAY(CLOSE, 1)?VOLUME:0), 26)*100
    fields = 'close, volume'
    offday = -27

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    condition_1 = closeData > delay_1
    sum_1 = dfSum(dfTripleOperation(condition_1, volData, 0), 26)
    condition_2 = -condition_1
    sum_2 = dfSum(dfTripleOperation(condition_2, volData, 0), 26) + 0.001
    result = sum_1 / sum_2 * 100

    return result.T[dateList]


def alpha_041(stockList, dateList):
    # (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
    fields = 'vwap'
    offday = -8

    vwpData = generateDataFrame(stockList, dateList, fields, offday)

    vwpDelta = dfDelay(vwpData, 3)
    vwpDeltaMax = dfMax(vwpDelta, 5)
    result = -1 * csRank(vwpDeltaMax)

    return result.T[dateList]


def alpha_042(stockList, dateList):
    # ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
    fields = 'high, volume'
    offday = -10

    highData, volData = generateDataFrame(stockList, dateList, fields, offday)
    highStd = dfSum(highData, 10)
    highRank = csRank(highStd)
    corrData = rollCorr(highData, volData, 10)
    result = -1 * highRank * corrData

    return result.T[dateList]


def alpha_043(stockList, dateList):
    # alpha_043 = SUM(CLOSE>DELAY(CLOSE,1)? VOLUME : ( CLOSE < DELAY(CLOSE ,1)? -VOLUME : 0),6)
    fields = 'close, volume'
    offday = -7

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay = dfDelay(closeData, 1)
    condition_1 = closeData > closeDelay
    condition_2 = closeData < closeDelay
    temp_1 = dfTripleOperation(condition_2, -1 * volData, 0)
    result = dfSum(dfTripleOperation(condition_1, volData, temp_1), 6)

    return result.T[dateList]


def alpha_044(stockList, dateList):
    # (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),
    # 3), 10), 15))
    fields = 'low  , volume  , vwap'
    offday = -28

    lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    volumeMean = dfMean(volData, 10)
    low_volumeCorr = rollCorr(lowData, volumeMean, 7)
    low_volumeCorrDecay = decayLinear(low_volumeCorr, 6)
    low_volume_tsRank = tsRank(low_volumeCorrDecay, 4)
    vwpDelta = dfDelta(vwapData, 3)
    vwpDecay = decayLinear(vwpDelta, 10)
    vwpRank = tsRank(vwpDecay, 15)
    result = low_volume_tsRank + vwpRank

    return result.T[dateList]


def alpha_045(stockList, dateList):
    # (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
    fields = 'open, close, volume, vwap'
    offday = -165

    openData, closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    close_open = closeData * 0.6 + openData * 0.4
    close_openDelta = dfDelta(close_open, 1)
    close_openDeltaRank = csRank(close_openDelta)
    volumeMean = dfMean(volData, 150)
    vwp_volumeCorr = rollCorr(vwapData, volumeMean, 15)
    vwp_volumeCorrRank = csRank(vwp_volumeCorr)
    result = close_openDeltaRank * vwp_volumeCorrRank

    return result.T[dateList]


def alpha_046(stockList, dateList):
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    fields = 'close'
    offday = -24

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    closeSum = dfMean(closeData, 3) + dfMean(closeData, 6) + dfMean(closeData, 12) + dfMean(closeData, 24)
    result = closeSum / (4 * closeData + 0.001)

    return result.T[dateList]


def alpha_047(stockList, dateList):
    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    fields = 'close, high,  low'
    offday = -15

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    highMax = dfMax(highData, 6)
    lowMin = dfMin(lowData, 6)
    numerator = highMax - closeData
    denumerator = highMax - lowMin + 0.001
    temp = numerator / denumerator * 100
    result = tsSma(temp, 9, 1)

    return result.T[dateList]


def alpha_048(stockList, dateList):
    # (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) +
    # SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
    fields = 'close, volume'
    offday = -20

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay_1 = dfDelay(closeData, 1)
    closeDelay_2 = dfDelay(closeData, 2)
    closeDelay_3 = dfDelay(closeData, 3)
    closeSign = dfSign(closeData - closeDelay_1) + dfSign(closeDelay_1 - closeDelay_2) + dfSign(
        closeDelay_2 - closeDelay_3)
    closeRank = csRank(closeSign)
    volumeSum_5 = dfSum(volData, 5)
    volumeSum_20 = dfSum(volData, 20) + 1
    result = -1 * closeRank * volumeSum_5 / volumeSum_20

    return result.T[dateList]


def alpha_049(stockList, dateList):
    # SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L
    # OW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L
    # OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI
    # GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    fields = 'high, low'
    offday = -13

    highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    highDelay = dfDelay(highData, 1)
    lowDelay = dfDelay(lowData, 1)

    conditon_1 = (highData + lowData) >= (highDelay + lowDelay)
    max_1 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_1 = dfTripleOperation(conditon_1, 0, max_1)
    sum_1 = dfSum(temp_1, 12)

    conditon_2 = (highData + lowData) >= (highDelay + lowDelay)
    max_2 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_2 = dfTripleOperation(conditon_2, 0, max_2)
    sum_2 = dfSum(temp_2, 12)

    conditon_3 = (highData + lowData) <= (highDelay + lowDelay)
    max_3 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_3 = dfTripleOperation(conditon_3, 0, max_3)
    sum_3 = dfSum(temp_3, 12)

    result = sum_1 / (sum_2 + sum_3 + 0.001)

    return result.T[dateList]


def alpha_050(stockList, dateList):
    # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L
    # OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L
    # OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI
    # GH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HI
    # GH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:
    # MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA
    # Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    fields = 'high, low'
    offday = -13

    highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    highDelay = dfDelay(highData, 1)
    lowDelay = dfDelay(lowData, 1)

    conditon_1 = (highData + lowData) <= (highDelay + lowDelay)
    max_1 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_1 = dfTripleOperation(conditon_1, 0, max_1)
    sum_1 = dfSum(temp_1, 12)

    conditon_2 = (highData + lowData) >= (highDelay + lowDelay)
    max_2 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_2 = dfTripleOperation(conditon_2, 0, max_2)
    sum_2 = dfSum(temp_2, 12)

    result = sum_1 / (sum_1 + sum_2 + 0.001) - sum_2 / (sum_1 + sum_2 + 0.001)

    return result.T[dateList]


def alpha_051(stockList, dateList):
    # SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L
    # OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L
    # OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI
    # GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    fields = 'high, low'
    offday = -13

    highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    highDelay = dfDelay(highData, 1)
    lowDelay = dfDelay(lowData, 1)

    conditon_1 = (highData + lowData) <= (highDelay + lowDelay)
    max_1 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_1 = dfTripleOperation(conditon_1, 0, max_1)
    sum_1 = dfSum(temp_1, 12)

    conditon_2 = (highData + lowData) >= (highDelay + lowDelay)
    max_2 = dfLarger(dfABS(highData - highDelay), dfABS(lowData - lowDelay))
    temp_2 = dfTripleOperation(conditon_2, 0, max_2)
    sum_2 = dfSum(temp_2, 12)

    result = sum_1 / (sum_1 + sum_2 + 0.001)

    return result.T[dateList]


def alpha_052(stockList, dateList):
    # alpha_052 =SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
    fields = 'close, high, low'
    offday = -27

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    mean_1 = (highData + lowData + closeData) / 3.0
    delay_1 = dfDelay(mean_1, 1)
    max_1 = dfLarger(0, highData - delay_1)
    sum_1 = dfSum(max_1, 26)
    max_2 = dfLarger(0, delay_1 - lowData)
    sum_2 = dfSum(max_2, 26) + 0.001
    result = sum_1 / sum_2

    return result.T[dateList]


def alpha_053(stockList, dateList):
    # alpha_053 =COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    fields = 'close'
    offday = -13

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay = dfDelay(closeData, 1)
    closeCount = dfCount((closeData > closeDelay), 12)
    result = closeCount / 12.0 * 100

    return result.T[dateList]


def alpha_054(stockList, dateList):
    # (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    fields = 'close , open'
    offday = -10

    closeData, openData = generateDataFrame(stockList, dateList, fields, offday)

    abs_1 = dfABS(closeData - openData)
    std_1 = dfStd(abs_1, 5)
    temp_1 = closeData - openData
    corr_1 = rollCorr(closeData, openData, 10)
    temp_2 = std_1 + temp_1 + corr_1
    result = -1 * csRank(temp_2)

    return result.T[dateList]


def alpha_055(stockList, dateList):
    '''
    SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CL
    OSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-
    DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-
    DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1)
    )?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))
    /4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY
    (CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    '''
    fields = 'high, close, open, low'
    offday = -13

    highData, closeData, openData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay = dfDelay(closeData, 1)
    openDelay = dfDelay(openData, 1)
    lowDelay = dfDelay(lowData, 1)
    A = 16 * (closeData - closeDelay + (closeData - openData) / 2.0 + openData - openDelay)

    high_closeDelay = dfABS(highData - closeDelay)
    low_closeDelay = dfABS(lowData - closeDelay)
    high_lowDelay = dfABS(highData - lowDelay)
    closeDelay_openDelay = dfABS(closeDelay - openDelay)

    B_condition_1 = np.logical_and((high_closeDelay > low_closeDelay), (high_closeDelay > high_lowDelay))
    B_first_1 = high_closeDelay + low_closeDelay / 2 + closeDelay_openDelay / 4.0
    B_condition_2 = np.logical_and((low_closeDelay > high_closeDelay), (low_closeDelay > high_closeDelay))
    B_first_2 = low_closeDelay + high_closeDelay / 2 + closeDelay_openDelay / 4.0
    B_second_2 = high_lowDelay + closeDelay_openDelay / 4.0
    B = dfTripleOperation(B_condition_1, B_first_1, dfTripleOperation(B_condition_2, B_first_2, B_second_2))
    
    C = dfLarger(high_closeDelay, low_closeDelay)

    result = dfSum(A / (B * C + 0.001), 12)

    return result.T[dateList]


def alpha_056(stockList, dateList):
    # (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),
    # SUM(MEAN(VOLUME,40), 19), 13))^5)))
    fields = 'high, volume, open, low'
    offday = -84

    highData, volumeData, openData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(openData - dfMin(openData, 12))
    sum_1 = dfSum((highData + lowData) / 2.0, 19)
    sum_2 = dfSum(dfMean(volumeData, 40), 19)
    corr_1 = rollCorr(sum_1, sum_2, 13)
    rank_2 = csRank(corr_1)
    tsRank_1 = tsRank(rank_2, 12)
    temp_1 = rank_1 < tsRank_1
    result = temp_1.applymap(lambda x: 1 if x else 0)

    return result.T[dateList]


def alpha_057(stockList, dateList):
    # SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    fields = 'high, close, low'
    offday = -12

    highData, closeData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    first = closeData - dfMin(lowData, 9)
    second = (dfMax(highData, 9) - dfMin(lowData, 9)) * 100 + 1
    result = tsSma(first / second, 3, 1)

    return result.T[dateList]


def alpha_058(stockList, dateList):
    # COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    fields = 'close'
    offday = -21

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay = dfDelay(closeData, 1)
    condition_1 = closeData > closeDelay
    count_1 = dfCount(condition_1, 20)
    result = count_1 / 20.0 * 100

    return result.T[dateList]


def alpha_059(stockList, dateList):
    # SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,D
    # ELAY(CLOSE,1)))),20)
    fields = 'close, high, low'
    offday = -21

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    condition_1 = closeData == delay_1
    condition_2 = closeData > delay_1

    temp_1 = dfTripleOperation(condition_1, 0, closeData - dfTripleOperation(condition_2, dfSmaller(lowData, delay_1),
                                                                             dfLarger(highData, delay_1)))
    result = dfSmaller(temp_1, 20)

    return result.T[dateList]


def alpha_060(stockList, dateList):
    # SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
    fields = 'close, high, low, volume'
    offday = -20

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (closeData - lowData) - (highData - closeData)
    temp_2 = (highData - lowData) * volData + 1
    result = dfSum(temp_1 / temp_2, 20)

    return result.T[dateList]


def alpha_061(stockList, dateList):
    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
    # RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
    fields = 'low, volume, vwap '
    offday = -105

    lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    decay_1 = decayLinear(dfDelta(vwapData, 1), 12)
    rank_1 = csRank(decay_1)
    decay_2 = decayLinear(csRank(rollCorr(lowData, dfMean(volData, 80), 8)), 17)
    rank_2 = csRank(decay_2)
    result = -1 * dfLarger(rank_1, rank_2)

    return result.T[dateList] 


def alpha_062(stockList, dateList):
    # (-1 * CORR(HIGH, RANK(VOLUME), 5))
    fields = 'high, volume'
    offday = -5

    highData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = -1 * rollCorr(highData, csRank(volData), 5)

    return result.T[dateList]


def alpha_063(stockList, dateList):
    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    fields = 'close'
    offday = -7

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = dfLarger(closeData - dfDelay(closeData, 1), 0)
    sma_1 = tsSma(temp_1, 6, 1)
    abs_1 = dfABS(closeData - dfDelay(closeData, 1))
    sma_2 = tsSma(abs_1, 6, 1) + 0.001
    result = sma_1 / sma_2 * 100

    return result.T[dateList]


def alpha_064(stockList, dateList):
    # (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
    # RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)

    fields = 'close, volume, vwap'
    offday = -91

    closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(csRank(vwapData), csRank(volData), 4)
    rank_1 = csRank(decayLinear(corr_1, 4))
    corr_2 = rollCorr(csRank(closeData), csRank(dfMean(volData, 60)), 4)
    rank_2 = csRank(decayLinear(tsRank(corr_2, 13), 14))
    result = -1 * dfSmaller(rank_1, rank_2)

    return result.T[dateList]


def alpha_065(stockList, dateList):
    # MEAN(CLOSE,6)/CLOSE
    fields = 'close'
    offday = -6

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfMean(closeData, 6) / (closeData + 0.001)

    return result.T[dateList]


def alpha_066(stockList, dateList):
    # (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    fields = 'close'
    offday = -6

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    mean_1 = dfMean(closeData, 6) + 0.001
    result = (closeData - mean_1) / mean_1 * 100

    return result.T[dateList]


def alpha_067(stockList, dateList):
    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    fields = 'close'
    offday = -25

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = dfLarger(closeData - dfDelay(closeData, 1), 0)
    sma_1 = tsSma(temp_1, 24, 1)
    abs_1 = dfABS(closeData - dfDelay(closeData, 1))
    sma_2 = tsSma(abs_1, 24, 1) + 0.001
    result = sma_1 / sma_2 * 100

    return result.T[dateList] 


def alpha_068(stockList, dateList):
    # SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    fields = 'high, low, volume'
    offday = -16

    highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (highData + lowData) / 2.0 - (dfDelay(highData, 1) + dfDelay(lowData, 1)) / 2.0
    result = tsSma(temp_1 * (highData - lowData) / (volData + 1), 15, 2)

    return result.T[dateList]


def alpha_069(stockList, dateList):
    # (SUM(DTM,20)>SUM(DBM,20)？ (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)： (SUM(DTM,20)=SUM(DBM,20)？
    # 0(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    fields = 'open, high, low'
    offday = -21

    openData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(openData, 1)
    delta_1 = dfDelta(openData, 1)
    DTM = dfTripleOperation(openData <= delay_1, 0, dfLarger(highData - openData, delta_1))
    DBM = dfTripleOperation(openData >= delay_1, 0, dfLarger(openData - lowData, delta_1))

    sum_1 = dfSum(DTM, 20) + 0.001
    sum_2 = dfSum(DBM, 20) + 0.001
    condition_1 = sum_1 > sum_2
    condition_2 = sum_1 == sum_2
    result = dfTripleOperation(condition_1, (sum_1 - sum_2) / sum_1,
                               dfTripleOperation(condition_2, 0, (sum_1 - sum_2) / sum_2))
    
    return result.T[dateList]


def alpha_070(stockList, dateList):
    # STD(AMOUNT,6)
    fields = 'amount'
    offday = -6

    amountData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfStd(amountData, 6)

    return result.T[dateList]


def alpha_071(stockList, dateList):
    # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    fields = 'close'
    offday = -24

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    mean_1 = dfMean(closeData, 24) + 0.001
    result = (closeData - mean_1) / mean_1 * 100

    return result.T[dateList]


def alpha_072(stockList, dateList):
    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    fields = 'close, high, low'
    offday = -21

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = dfMax(highData, 6) - closeData
    temp_2 = dfMax(highData, 6) - dfMin(lowData, 6) + 0.001
    result = tsSma(temp_1 / temp_2 * 100, 15, 1)

    return result.T[dateList]


def alpha_073(stockList, dateList):
    # ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -
    # RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
    fields = 'close, volume, vwap'
    offday = -37

    closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    decay_1 = decayLinear(decayLinear(rollCorr(closeData, volData, 10), 16), 4)
    rank_1 = tsRank(decay_1, 5)
    rank_2 = csRank(decayLinear(rollCorr(vwapData, dfMean(volData, 30), 4), 3))
    result = -1 * (rank_1 - rank_2)

    return result.T[dateList] 


def alpha_074(stockList, dateList):
    # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20),
    # SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
    fields = 'low, volume, vwap'
    offday = -67

    lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    sum_1 = dfSum(lowData * 0.35 + vwapData * 0.65, 20)
    sum_2 = dfSum(dfMean(volData, 40), 20)
    rank_1 = csRank(rollCorr(sum_1, sum_2, 7))
    rank_2 = csRank(rollCorr(csRank(vwapData), csRank(volData), 6))
    result = rank_1 + rank_2

    return result.T[dateList]


def alpha_075(stockList, dateList):
    # COUNT(CLOSE>OPEN & MARKETCLOSE<MARKETOPEN,50) / COUNT(MARKETCLOSE<BMARKETOPEN,50)
    fields = 'open,close'
    offday = -50

    openData, closeData = generateDataFrame(stockList, dateList, fields, offday)
    marketOpen, marketClose = openData.mean(axis=1), closeData.mean(axis=1)
 
    condition_1 = dfAND(closeData > openData, (marketClose < marketOpen).to_frame())
    condition_2 = (marketClose < marketOpen).to_frame()
    result = np.divide(dfCount(condition_1, 50), (dfCount(condition_2, 50) + 1))

    return result.T[dateList]


def alpha_076(stockList, dateList):
    # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    fields = 'close, volume'
    offday = -21

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1) + 0.001
    temp_1 = (dfABS(closeData / delay_1) - 1) / (volData + 1) + 1
    result = dfStd(temp_1, 20) / dfMean(temp_1, 20)

    return result.T[dateList] 


def alpha_077(stockList, dateList):
    # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),
    #         RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
    fields = 'high, low, volume, vwap'
    offday = -49

    highData, lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear((highData + lowData) / 2.0 + highData - (vwapData + highData), 20))
    rank_2 = csRank(decayLinear(rollCorr((highData + lowData) / 2.0, dfMean(volData, 40), 3), 6))
    result = dfSmaller(rank_1, rank_2)

    return result.T[dateList] 


def alpha_078(stockList, dateList):
    # ((HIGH+LOW+CLOSE)/3-MEAN((HIGH+LOW+CLOSE)/3,12))/
    # (0.0015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    fields = 'close,  high, low'
    offday = -24

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (highData + lowData + closeData) / 3.0
    mean_1 = dfMean(temp_1, 12)
    result = (temp_1 - mean_1) / (0.0015 * dfMean(dfABS(closeData - mean_1) + 0.001, 12))

    return result.T[dateList]


def alpha_079(stockList, dateList):
    # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    fields = 'close'
    offday = -13

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delta_1 = dfDelta(closeData, 1)
    sma_1 = tsSma(dfLarger(delta_1, 0), 12, 1)
    sma_2 = tsSma(dfABS(delta_1), 12, 1) + 0.001
    result = sma_1 / sma_2 * 100

    return result.T[dateList] 


def alpha_080(stockList, dateList):
    # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    fields = 'volume'
    offday = -5

    volData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(volData, 5) / (dfDelay(volData, 5) + 1) * 100

    return result.T[dateList]


def alpha_081(stockList, dateList):
    # SMA(VOLUME,21,2)
    fields = 'volume'
    offday = -21

    volData = generateDataFrame(stockList, dateList, fields, offday)

    result = tsSma(volData, 21, 2)

    return result.T[dateList]


def alpha_082(stockList, dateList):
    # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    fields = 'close, high, low'
    offday = -26

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = dfMax(highData, 6) - closeData
    temp_2 = dfMax(highData, 6) - dfMin(lowData, 6) + 0.001
    result = tsSma(temp_1 / temp_2 * 100, 20, 1)

    return result.T[dateList]


def alpha_083(stockList, dateList):
    # (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
    fields = 'high, volume'
    offday = -5

    highData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = -1 * csRank(rollCorr(csRank(highData), csRank(volData), 5))

    return result.T[dateList] 


def alpha_084(stockList, dateList):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
    fields = 'close, volume'
    offday = -30

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    condition_1 = closeData > delay_1
    condition_2 = closeData < delay_1
    temp_1 = dfTripleOperation(condition_1, volData, dfTripleOperation(condition_2, -1 * volData, 0))
    result = dfSum(temp_1, 20)

    return result.T[dateList]


def alpha_085(stockList, dateList):
    # (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
    fields = 'close, volume'
    offday = -40

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    # 进行运算
    rank_1 = tsRank(volData / dfMean(volData + 1, 20), 20)
    rank_2 = tsRank(-1 * dfDelta(closeData, 7), 8)
    result = rank_1 * rank_2

    return result.T[dateList]


def alpha_086(stockList, dateList):
    # ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) -
    #     ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :
    #     (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) -
    #     ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) *
    #     (CLOSE - DELAY(CLOSE, 1)))))
    fields = 'close'
    offday = -20

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 10)
    delay_2 = dfDelay(closeData, 20)
    temp_1 = (delay_2 - delay_1) / 10.0 - (delay_1 - closeData) / 10.0
    condition_1 = 0.25 < temp_1
    condition_2 = temp_1 < 0
    result = dfTripleOperation(condition_1, -1, dfTripleOperation(condition_2, 1, -1 * dfDelta(closeData, 1)))

    return result.T[dateList] 


def alpha_087(stockList, dateList):
    # ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9)
    # + (HIGH * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
    fields = 'open, close, high, low, vwap'
    offday = -18

    openData, closeData, highData, lowData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear(dfDelta(vwapData, 4), 7))
    temp_1 = (lowData * 0.9 + highData * 0.1- vwapData) / (openData - (highData + lowData) / 2.0 + 0.001)
    rank_2 = tsRank(decayLinear(temp_1, 11), 7)
    result = -1 * (rank_1 + rank_2)

    return result.T[dateList]


def alpha_088(stockList, dateList):
    # (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    fields = 'close'
    offday = -20

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(closeData, 20) / (dfDelay(closeData, 20) + 0.001) * 100

    return result.T[dateList]


def alpha_089(stockList, dateList):
    # 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    fields = 'close'
    offday = -37

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(closeData, 13, 2)
    sma_2 = tsSma(closeData, 27, 2)
    sma_3 = tsSma(sma_1 - sma_2, 10, 2)
    result = 2 * (sma_1 - sma_2 - sma_3)

    return result.T[dateList] 


def alpha_090(stockList, dateList):
    # ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
    fields = 'volume, vwap'
    offday = -5

    volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    result = -1 * csRank(rollCorr(csRank(vwapData), csRank(volData), 5))

    return result.T[dateList]


def alpha_091(stockList, dateList):
    # ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
    fields = 'close, low, volume'
    offday = -45

    closeData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(closeData - dfMax(closeData, 5))
    rank_2 = csRank(rollCorr(dfMean(volData, 40), lowData, 5))
    result = -1 * rank_1 * rank_2

    return result.T[dateList]


def alpha_092(stockList, dateList):
    # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),
    # TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)
    fields = 'close, vwap, volume'
    offday = -220

    closeData, vwapData, volData = generateDataFrame(stockList, dateList, fields, offday)

    close_vwap = closeData * 0.35 + vwapData * 0.65
    delta_1 = dfDelta(close_vwap, 2)
    decay_1 = decayLinear(delta_1, 3)
    rank_1 = csRank(decay_1)
    corr_1 = rollCorr(dfMean(volData, 180), closeData, 13)
    decay_2 = decayLinear(dfABS(corr_1), 5)
    tsRank_1 = tsRank(decay_2, 15)
    max_1 = dfLarger(rank_1, tsRank_1)
    result = -1 * max_1

    return result.T[dateList]


def alpha_093(stockList, dateList):
    # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    fields = 'open, low'
    offday = -21

    openData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    openDelay = dfDelay(openData, 1)
    condition_1 = openData >= openDelay
    max_1 = dfLarger(openData - lowData, openData - openDelay)
    triple_1 = dfTripleOperation(condition_1, 0, max_1)
    result = dfSum(triple_1, 20)

    return result.T[dateList]


def alpha_094(stockList, dateList):
    # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    fields = 'close, volume'
    offday = -31

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay_1 = dfDelay(closeData, 1)
    condition_1 = closeData > closeDelay_1
    condition_2 = closeData < closeDelay_1
    temp_1 = dfTripleOperation(condition_1, volData, dfTripleOperation(condition_2, -1 * volData, 0))
    result = dfSum(temp_1, 30)

    return result.T[dateList]


def alpha_095(stockList, dateList):
    # STD(AMOUNT,20)
    fields = 'amount'
    offday = -20

    amountData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfStd(amountData, 20)

    return result.T[dateList]


def alpha_096(stockList, dateList):
    # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    fields = 'high, low, close'
    offday = -15

    highData, lowData, closeData = generateDataFrame(stockList, dateList, fields, offday)

    first_1 = closeData - dfMin(lowData, 9)
    second_1 = dfMax(highData, 9) - dfMin(lowData, 9) + 0.001
    temp_1 = first_1 / second_1 * 100
    sma_1 = tsSma(temp_1, 3, 1)
    result = tsSma(sma_1, 3, 1)

    return result.T[dateList]


def alpha_097(stockList, dateList):
    # STD(VOLUME,10)
    fields = 'volume'
    offday = -10

    volData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfStd(volData, 10)

    return result.T[dateList]


def alpha_098(stockList, dateList):
    # ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /
    # DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
    fields = 'close'
    offday = -200

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delta_1 = dfDelta(dfSum(closeData, 100) / 100.0, 100)
    first_1 = delta_1 / dfDelay(closeData, 100)
    condition_1 = first_1 <= 0.05
    a_1 = -1 * (closeData - dfMin(closeData, 100))
    b_1 = -1 * (dfDelta(closeData, 3))

    result = dfTripleOperation(condition_1, a_1, b_1)

    return result.T[dateList]


def alpha_099(stockList, dateList):
    # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    fields = 'close, volume'
    offday = -5

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    cov_1 = rollCov(csRank(closeData), csRank(volData), 5)
    result = -1 * csRank(cov_1)

    return result.T[dateList]


def alpha_100(stockList, dateList):
    # STD(VOLUME,20)
    fields = 'volume'
    offday = -20

    volData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfStd(volData, 20)

    return result.T[dateList]


def alpha_101(stockList, dateList):
    # ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
    # RANK(VOLUME), 11))) * -1)
    fields = 'close, high, volume, vwap'
    offday = -82

    closeData, highData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(closeData, dfSum(dfMean(volData, 30), 37), 15)
    rank_1 = csRank(corr_1)
    rank_2 = csRank(highData * 0.1 + vwapData * 0.9)
    rank_3 = csRank(volData)
    corr_2 = rollCorr(rank_2, rank_3, 11)
    rank_4 = csRank(corr_2)
    temp_1 = rank_1 < rank_4
    result = temp_1.applymap(lambda x: -1 if x else 0)

    return result.T[dateList]


def alpha_102(stockList, dateList):
    # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    fields = 'volume'
    offday = -7

    volData = generateDataFrame(stockList, dateList, fields, offday)

    delta_1 = dfDelta(volData, 1)
    max_1 = dfLarger(delta_1, 0)
    sma_1 = tsSma(max_1, 6, 1)
    abs_1 = dfABS(delta_1) + 1
    sma_2 = tsSma(abs_1, 6, 1)
    result = sma_1 / sma_2 * 100

    return result.T[dateList]


def alpha_103(stockList, dateList):
    # ((20-LOWDAY(LOW,20))/20)*100
    fields = 'low'
    offday = -20

    lowData = generateDataFrame(stockList, dateList, fields, offday)

    result = (20 - lowDay(lowData, 20)) / 20.0 * 100

    return result.T[dateList]

def alpha_104(stockList, dateList):
    # (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    fields = 'close, high, volume'
    offday = -20

    closeData, highData, volData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(highData, volData, 5)
    delta_1 = dfDelta(corr_1, 5)
    rank_1 = csRank(dfStd(closeData, 20))
    result = -1 * delta_1 * rank_1

    return result.T[dateList]


def alpha_105(stockList, dateList):
    # (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    fields = 'open, volume'
    offday = -10

    openData, volData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(csRank(openData), csRank(volData), 10)
    result = -1 * corr_1

    return result.T[dateList]


def alpha_106(stockList, dateList):
    # CLOSE-DELAY(CLOSE,20)
    fields = 'close'
    offday = -20

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(closeData, 20)

    return result.T[dateList]


def alpha_107(stockList, dateList):
    # (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
    fields = 'open, close, high, low '
    offday = -1

    openData, closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(openData - dfDelay(highData, 1))
    rank_2 = csRank(openData - dfDelay(closeData, 1))
    rank_3 = csRank(openData - dfDelay(lowData, 1))
    result = -1 * rank_1 * rank_2 * rank_3

    return result.T[dateList]


def alpha_108(stockList, dateList):
    # ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
    fields = 'high, volume, vwap'
    offday = -126

    highData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(highData - dfMin(highData, 2))
    rank_2 = csRank(rollCorr(vwapData, dfMean(volData, 120), 6))
    result = -1 * (rank_1 ** rank_2)

    return result.T[dateList]


def alpha_109(stockList, dateList):
    # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    fields = 'high, low'
    offday = -20

    highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(highData - lowData, 10, 2)
    sma_2 = tsSma(sma_1, 10, 2) + 0.001
    result = sma_1 / sma_2

    return result.T[dateList]


def alpha_110(stockList, dateList):
    # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    fields = 'close, high, low'
    offday = -21

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    max_1 = dfLarger(0, highData - dfDelay(closeData, 1))
    sum_1 = dfSum(max_1, 20)
    max_2 = dfLarger(0, dfDelay(closeData, 1) - lowData)
    sum_2 = dfSum(max_2, 20) + 0.001
    result = sum_1 / sum_2 * 100

    return result.T[dateList]


def alpha_111(stockList, dateList):
    # SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    fields = 'close, high, low, volume'
    offday = -11

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = volData * ((closeData - lowData) - (highData - closeData)) / (highData - lowData + 0.001)
    sam_1 = tsSma(temp_1, 11, 2)
    sam_2 = tsSma(temp_1, 4, 2) + 0.001
    result = sam_1 / sam_2

    return result.T[dateList]


def alpha_112(stockList, dateList):
    # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE
    # -DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DE
    # LAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    fields = 'close'
    offday = -13

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = closeData - dfDelay(closeData, 1)
    condition_1 = temp_1 > 0
    sum_1 = dfSum(dfTripleOperation(condition_1, temp_1, 0), 12)
    condition_2 = temp_1 < 0
    sum_2 = dfSum(dfTripleOperation(condition_2, dfABS(temp_1), 0), 12)
    sum_11 = sum_1 - sum_2
    sum_22 = sum_1 + sum_2 + 0.001
    result = sum_11 / sum_22 * 100

    return result.T[dateList]


def alpha_113(stockList, dateList):
    # (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))
    fields = 'close, volume'
    offday = -25

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(dfSum(dfDelay(closeData, 5), 20) / 20.0)
    corr_1 = rollCorr(closeData, volData, 2)
    rank_2 = csRank(rollCorr(dfSum(closeData, 5), dfSum(closeData, 20), 2))
    result = -1 * rank_1 * corr_1 * rank_2

    return result.T[dateList]


def alpha_114(stockList, dateList):
    # ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    fields = 'close, high, low, volume, vwap'
    offday = -7

    closeData, highData, lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(dfDelay((highData - lowData) / (dfSum(closeData, 5) / 5.0), 2))
    rank_2 = csRank(volData)
    denumerator_1 = (highData - lowData) / (dfSum(closeData, 5) / 5.0) / (vwapData - closeData + 0.001)
    result = rank_1 * rank_2 / (denumerator_1 + 0.001)
    
    return result.T[dateList]


def alpha_115(stockList, dateList):
    # (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))
    fields = 'close, high, low, volume'
    offday = -40

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(rollCorr(highData * 0.9 + closeData * 0.1, dfMean(volData, 30), 10))
    rank_2 = csRank(rollCorr(tsRank((highData + lowData) / 2.0, 4), tsRank(volData, 10), 7))
    result = rank_1 ** rank_2

    return result.T[dateList]


def alpha_116(stockList, dateList):
    # REGBETA(CLOSE,SEQUENCE,20)
    fields = 'close'
    offday = -20

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfREGBETA(closeData, np.arange(1, 21), 20)
    
    return result.T[dateList]


def alpha_117(stockList, dateList):
    # ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
    fields = 'close, high, low, volume, p_change '
    offday = -32

    closeData, highData, lowData, volData, retData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = tsRank(volData, 32)
    temp_1 = 1 - tsRank(closeData + highData - lowData, 16)
    temp_2 = 1 - tsRank(retData, 32)
    result = rank_1 * temp_1 * temp_2

    return result.T[dateList]


def alpha_118(stockList, dateList):
    # SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    fields = 'open, high, low'
    offday = -20

    openData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    sum_1 = dfSum(highData - openData, 20)
    sum_2 = dfSum(openData - lowData, 20) + 0.001
    result = sum_1 / sum_2 * 100

    return result.T[dateList]


def alpha_119(stockList, dateList):
    # (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) -
    # RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
    fields = 'open, volume, vwap'
    offday = -60

    openData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    corr_1 = rollCorr(vwapData, dfSum(dfMean(volData, 5), 26), 5)
    rank_1 = csRank(decayLinear(corr_1, 7))

    corr_2 = rollCorr(csRank(openData), csRank(dfMean(volData, 15)), 21)
    rank_2 = tsRank(dfMin(corr_2, 9), 7)
    rank_3 = csRank(decayLinear(rank_2, 8))
    result = (rank_1 - rank_3) * 100

    return result.T[dateList]


def alpha_120(stockList, dateList):
    # (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))预处理数据
    fields = 'close, vwap'
    offday = -0

    closeData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    result = csRank(vwapData - closeData) / csRank(vwapData + closeData)

    return result.T[dateList]


def alpha_121(stockList, dateList):
    # ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)
    fields = 'volume ,vwap'
    offday = -83

    volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(vwapData - dfMin(vwapData, 12))
    corr_1 = rollCorr(tsRank(vwapData, 20), tsRank(dfMean(volData, 60), 2), 18)
    rank_2 = tsRank(corr_1, 3)
    result = -1 * rank_1 ** rank_2

    return result.T[dateList]


def alpha_122(stockList, dateList):
    # (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),
    # 13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(tsSma(tsSma(dfLog(closeData), 13, 2), 13, 2), 13, 2)
    delay_1 = dfDelay(sma_1, 1) + 0.001
    result = (sma_1 - delay_1) / delay_1

    return result.T[dateList]


def alpha_123(stockList, dateList):
    # ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
    fields = 'high, low, volume'
    offday = -90

    highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    sum_1 = dfSum((highData + lowData) / 2.0, 20)
    sum_2 = dfSum(dfMean(volData, 60), 20)
    rank_1 = csRank(rollCorr(sum_1, sum_2, 9))
    rank_2 = csRank(rollCorr(lowData, volData, 6))
    temp_1 = rank_1 < rank_2
    result = temp_1.applymap(lambda x: -1 if x else 0)

    return result.T[dateList]


def alpha_124(stockList, dateList):
    # (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
    fields = 'close, vwap'
    offday = -32

    closeData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = closeData - vwapData
    temp_2 = decayLinear(csRank(dfMax(closeData, 30)), 2) + 0.001
    result = temp_1 / temp_2

    return result.T[dateList]


def alpha_125(stockList, dateList):
    # (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) /
    # RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))
    fields = 'close, volume, vwap'
    offday = -117

    closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    decay_1 = decayLinear(rollCorr(vwapData, dfMean(volData, 80), 17), 20)
    rank_1 = csRank(decay_1)
    decay_2 = decayLinear(dfDelta(closeData * 0.5 + vwapData * 0.5, 3), 16)
    rank_2 = csRank(decay_2)
    result = rank_1 / rank_2

    return result.T[dateList] 


def alpha_126(stockList, dateList):
    # (CLOSE+HIGH+LOW)/3
    fields = 'close, high, low'
    offday = 0

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    result = (closeData + highData + lowData) / 3.0

    return result.T[dateList]


def alpha_127(stockList, dateList):
    # (MEAN((100*(CLOSE-TSMAX(CLOSE,12))/(TSMAX(CLOSE,12)))^2, 12))^(1/2)
    fields = 'close'
    offday = -24

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = 100 * (closeData - dfMax(closeData, 12)) / (dfMax(closeData, 12) + 0.001)
    result = dfMean(temp_1 ** 2, 12) ** 0.5

    return result.T[dateList] 


def alpha_128(stockList, dateList):
    # 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?
    # (HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?
    # (HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    fields = 'close, high, low, volume'
    offday = -20

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (highData + lowData + closeData) / 3.0
    delay_1 = dfDelay(temp_1, 1)
    condition_1 = temp_1 > delay_1
    condition_2 = temp_1 < delay_1
    sum_1 = dfSum(dfTripleOperation(condition_1, temp_1 * volData, 0), 14) + 1
    sum_2 = dfSum(dfTripleOperation(condition_2, temp_1 * volData, 0), 14) + 1
    result = 100 - (100 / (1 + sum_1 / sum_2))

    return result.T[dateList] 


def alpha_129(stockList, dateList):
    # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    fields = 'close'
    offday = -13

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delta_1 = dfDelta(closeData, 1)
    condition_1 = delta_1 < 0
    result = dfSum(dfTripleOperation(condition_1, dfABS(delta_1), 0), 12)

    return result.T[dateList]


def alpha_130(stockList, dateList):
    # (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) /
    # RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))
    fields = 'high, low, volume, vwap'
    offday = -60

    highData, lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear(rollCorr((highData + lowData) / 2.0, dfMean(volData, 40), 9), 10))
    rank_2 = csRank(decayLinear(rollCorr(csRank(vwapData), csRank(volData), 7), 3))
    result = rank_1 / rank_2

    return result.T[dateList]


def alpha_131(stockList, dateList):
    # (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
    fields = 'close, volume,  vwap'
    offday = -86

    closeData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(dfDelta(vwapData, 1))
    rank_2 = tsRank(rollCorr(closeData, dfMean(volData, 50), 18), 18)
    result = rank_1 ** rank_2

    return result.T[dateList] 


def alpha_132(stockList, dateList):
    # MEAN(AMOUNT,20)
    fields = 'amount'
    offday = -20

    amountData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfMean(amountData, 20)

    return result.T[dateList]


def alpha_133(stockList, dateList):
    # ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    fields = 'high, low'
    offday = -20

    highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    result = (20 - highDay(highData, 20)) / 20.0 * 100 - (20 - lowDay(lowData, 20)) / 20.0 * 100

    return result.T[dateList] 


def alpha_134(stockList, dateList):
    # (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    fields = 'close, volume'
    offday = -12

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(closeData, 12) / (dfDelay(closeData, 12) + 0.001) * volData

    return result.T[dateList]


def alpha_135(stockList, dateList):
    # SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    fields = 'close'
    offday = -42

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = tsSma(dfDelay(closeData / (dfDelay(closeData, 20) + 0.001), 1), 20, 1)

    return result.T[dateList]


def alpha_136(stockList, dateList):
    # ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    fields = 'open, volume, p_change'
    offday = -20

    openData, volData, retData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(dfDelta(retData, 3))
    result = -1 * rank_1 * rollCorr(openData, volData, 10)

    return result.T[dateList]


def alpha_137(stockList, dateList):
    # 16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,
    # 1))>ABS(LOW-DELAY(CLOSE,1)) &
	# ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS
	# E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) &
	# ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLO
	# SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP
	#EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    fields = 'open, close, high, low'
    offday = -1

    openData, closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    closeDelay = dfDelay(closeData, 1)
    openDelay = dfDelay(openData, 1)
    lowDelay = dfDelay(lowData, 1)
    temp_1 = closeData - closeDelay + (closeData - openData) / 2.0 + closeDelay - openDelay
    high_close = dfABS(highData - closeDelay)
    low_close = dfABS(lowData - closeDelay)
    high_low = dfABS(highData - lowDelay)
    close_open = dfABS(closeDelay - openDelay)
    condition_1 = high_close > low_close
    condition_2 = high_close > high_low
    condition_11 = dfAND(condition_1, condition_2)
    a_1 = high_close + low_close / 2.0 + close_open / 4.0
    condition_3 = low_close > high_low
    condition_4 = low_close > high_close
    condition_22 = dfAND(condition_3, condition_4)
    a_2 = low_close + high_close / 2.0 + close_open / 4.0
    b_1 = high_low + close_open / 4.0
    temp_2 = dfTripleOperation(condition_11, a_1, dfTripleOperation(condition_22, a_2, b_1)) + 1
    temp_3 = dfLarger(high_close, low_close)

    result = 16 * temp_1 / temp_2 * temp_3

    return result.T[dateList]


def alpha_138(stockList, dateList):
    # ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) -
    # TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)
    fields = 'low, volume, vwap'
    offday = -124

    lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear(dfDelta(lowData * 0.7 + vwapData * 0.3, 3), 20))
    rank_2 = tsRank(dfMean(volData, 60), 17)
    rank_3 = tsRank(lowData, 8)
    rank_4 = tsRank(rollCorr(rank_3, rank_2, 5), 19)
    rank_5 = tsRank(decayLinear(rank_4, 16), 7)
    result = -1 * (rank_1 - rank_5)

    return result.T[dateList]


def alpha_139(stockList, dateList):
    # (-1 * CORR(OPEN, VOLUME, 10))
    fields = 'open, volume'
    offday = -10

    openData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = -1 * rollCorr(openData, volData, 10)

    return result.T[dateList]


def alpha_140(stockList, dateList):
    # MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)),
    # TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))
    fields = 'open, close, high, low, volume'
    offday = -98

    openData, closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear(csRank(openData) + csRank(lowData) - csRank(highData) - csRank(closeData), 8))
    rank_2 = tsRank(dfMean(volData, 60), 20)
    rank_3 = tsRank(decayLinear(rollCorr(tsRank(closeData, 8), rank_2, 8), 7), 3)
    result = dfSmaller(rank_1, rank_3)

    return result.T[dateList]


def alpha_141(stockList, dateList):
    # (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
    fields = 'high, volume'
    offday = -24

    highData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = -1 * csRank(rollCorr(csRank(highData), csRank(dfMean(volData, 15)), 9))

    return result.T[dateList]


def alpha_142(stockList, dateList):
    # (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))
    fields = 'close, volume'
    offday = -25

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(tsRank(closeData, 10))
    rank_2 = csRank(dfDelta(dfDelta(closeData, 1), 1))
    rank_3 = csRank(tsRank(volData / (dfMean(volData, 20) + 1), 5))
    result = -1 * rank_1 * rank_2 * rank_3

    return result.T[dateList]


def alpha_143(stockList, dateList):
    # CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    fields = 'close'
    offday = -1

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = closeData > dfDelay(closeData, 1)
    result = dfTripleOperation(condition_1, dfDelta(closeData, 1) / dfDelay(closeData, 1), 1)

    return result.T[dateList]


def alpha_144(stockList, dateList):
    # SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    fields = 'close, amount'
    offday = -21

    closeData, amountData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    condition_1 = closeData < delay_1
    temp_1 = dfSumif(dfABS(closeData / delay_1 - 1) / amountData, 20, condition_1)
    temp_2 = dfCount(condition_1, 20) + 1.0
    result = temp_1 / temp_2

    return result.T[dateList]


def alpha_145(stockList, dateList):
    # (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    fields = 'volume'
    offday = -26

    volData = generateDataFrame(stockList, dateList, fields, offday)

    result = (dfMean(volData, 9) - dfMean(volData, 26)) / (dfMean(volData, 12) + 1) * 100

    return result.T[dateList]


def alpha_146(stockList, dateList):
    # MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((
    # CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOS
    # E-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,
    # 1))/DELAY(CLOSE,1),61,2)))^2,60)
    fields = 'p_change'
    offday = -150

    pchData = generateDataFrame(stockList, dateList, fields, offday)

    SMApch = tsSma(pchData, 61, 2)
    temp_1 = pchData - SMApch
    temp_2 = dfMean(temp_1, 20) * temp_1
    temp_3 = tsSma((pchData - temp_1) ** 2, 60, 2) + 0.001
    result = temp_1 * temp_2 / temp_3

    return result.T[dateList]


def alpha_147(stockList, dateList):
    # REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    fields = 'close'
    offday = -24

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfREGBETA(dfMean(closeData, 12), np.arange(1, 13), 12)

    return result.T[dateList]


def alpha_148(stockList, dateList):
    # ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    fields = 'open, volume'
    offday = -75

    openData, volData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(rollCorr(openData, dfSum(dfMean(volData, 60), 9), 6))
    rank_2 = csRank(openData - dfMin(openData, 14))
    temp_1 = rank_1 < rank_2
    result = temp_1.applymap(lambda x: -1 if x else 0)

    return result.T[dateList] 


def alpha_149(stockList, dateList):
    # REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)
    # ),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELA
    # Y(BANCHMARKINDEXCLOSE,1)),252)
    fields = 'close'
    offday = -250

    closeData = generateDataFrame(stockList, dateList, fields, offday)
    marketClose = closeData.mean(axis=1)
    index = closeData.index
    delay_1 = dfDelay(closeData, 1)
    delay_2 = dfDelay(marketClose, 1)
    condition_1  = marketClose < delay_2
    filter_1 = dfFilter(closeData / delay_1 - 1, condition_1)
    filter_2 = dfFilter(marketClose / delay_2 - 1, condition_1).to_frame()
    sr_1 = filter_2.iloc[:, 0]
    jointDf = pd.concat([filter_1, sr_1], axis=1, join='inner')
    df_1 = jointDf.iloc[:, :-1]
    sr_2 = jointDf.iloc[:, -1]
    length = len(jointDf) - len(dateList)
    result = dfREGBETA(df_1, sr_2, length)
    result = result.reindex(index, method='pad')
    result.columns.name='code'

    return result.T[dateList]


def alpha_150(stockList, dateList):
    # (CLOSE+HIGH+LOW)/3*VOLUME
    fields = 'close, high, low, volume'
    offday = 0

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = (closeData + highData + lowData) / 3.0 * volData

    return result.T[dateList]


def alpha_151(stockList, dateList):
    # SMA(CLOSE-DELAY(CLOSE,20),20,1)
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = tsSma(dfDelta(closeData, 20), 20, 1)

    return result.T[dateList]


def alpha_152(stockList, dateList):
    # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY
    # (CLOSE,9),1),9,1),1),26),9,1)
    fields = 'close'
    offday = -55

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(dfDelay(closeData / (dfDelay(closeData, 9) + 0.001), 1), 9, 1)
    mean_1 = dfMean(dfDelay(sma_1, 1), 12)
    mean_2 = dfMean(dfDelay(sma_1, 1), 26)
    result = tsSma(mean_1 - mean_2, 9, 1)

    return result.T[dateList]


def alpha_153(stockList, dateList):
    # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    fields = 'close'
    offday = -24

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    mean_1 = dfMean(closeData, 3) + dfMean(closeData, 6) + dfMean(closeData, 12) + dfMean(closeData, 24)
    result = mean_1 / 4.0

    return result.T[dateList]


def alpha_154(stockList, dateList):
    # (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
    fields = 'volume, vwap'
    offday = -198

    volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = vwapData - dfMin(vwapData, 16)
    temp_2 = rollCorr(vwapData, dfMean(volData, 180), 18)
    temp_3 = temp_1 < temp_2
    result = temp_3.applymap(lambda x: 1 if x else 0)

    return result.T[dateList]


def alpha_155(stockList, dateList):
    # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    fields = 'volume'
    offday = -37

    volData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(volData, 13, 2)
    sma_2 = tsSma(volData, 27, 2)
    sma_3 = tsSma(sma_1 - sma_2, 10, 2)
    result = sma_1 - sma_2 - sma_3

    return result.T[dateList]


def alpha_156(stockList, dateList):
    # (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),
    # 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)
    fields = 'open, low, vwap'
    offday = -8

    openData, lowData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(decayLinear(dfDelta(vwapData, 5), 3))
    delta_1 = dfDelta(openData * 0.15 + lowData * 0.85, 2)
    rank_2 = csRank(decayLinear(-1 * delta_1 / (openData * 0.15 + lowData * 0.85 + 0.001), 3))
    result = -1 * dfLarger(rank_1, rank_2)

    return result.T[dateList]


def alpha_157(stockList, dateList):
    # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) +
    # TSRANK(DELAY((-1 * RET), 6), 5))
    fields = 'close, p_change'
    offday = -15

    closeData, retData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(-1 * csRank(dfDelta(closeData - 1, 5)))
    min_1 = dfMin(rank_1, 2)
    rank_2 = csRank(dfLog(dfSum(min_1, 1) + 1))
    min_2 = dfMin(dfProd(rank_2, 1), 5)
    result = min_2 + tsRank(dfDelay(-1 * retData, 6), 5)

    return result.T[dateList]


def alpha_158(stockList, dateList):
    # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    fields = 'close, high, low'
    offday = -15

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = highData - tsSma(closeData, 15, 2) - (lowData - tsSma(closeData, 15, 2))
    result = temp_1 / (closeData + 0.001)

    return result.T[dateList]


def alpha_159(stockList, dateList):
    # ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)
    # *12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CL
    # OSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D
    # ELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    fields = 'close, high, low'
    offday = -25

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    temp_1 = closeData - dfSum(dfSmaller(lowData, delay_1), 6)
    sum_1 = dfSum(dfLarger(highData, delay_1) - dfSmaller(lowData, delay_1), 6) + 0.001
    temp_11 = temp_1 / sum_1 * 12 * 24

    temp_2 = closeData - dfSum(dfSmaller(lowData, delay_1), 12)
    sum_2 = dfSum(dfLarger(highData, delay_1) - dfSmaller(lowData, delay_1), 12) + 0.001
    temp_22 = temp_2 / sum_2 * 6 * 24

    temp_3 = closeData - dfSum(dfSmaller(lowData, delay_1), 24)
    sum_3 = dfSum(dfLarger(highData, delay_1) - dfSmaller(lowData, delay_1), 24) + 0.001
    temp_33 = temp_3 / sum_3 * 6 * 24

    result = (temp_11 + temp_22 + temp_33) * 100 / (6.0 * 12 + 6 * 24 + 12 * 24)

    return result.T[dateList]


def alpha_160(stockList, dateList):
    # SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = closeData <= dfDelay(closeData, 1)
    result = tsSma(dfTripleOperation(condition_1, dfStd(closeData, 20), 0), 20, 1)

    return result.T[dateList]


def alpha_161(stockList, dateList):
    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    fields = 'close, high, low'
    offday = -13

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(closeData, 1)
    abs_1 = dfABS(delay_1 - highData)
    abs_2 = dfABS(delay_1 - lowData)
    max_1 = dfLarger(highData - lowData, abs_1)
    max_2 = dfLarger(max_1, abs_2)
    result = dfMean(max_2, 12)

    return result.T[dateList]


def alpha_162(stockList, dateList):
    # (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOS
    # E-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(C
    # LOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,
    # 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    fields = 'close'
    offday = -25

    closeData = generateDataFrame(stockList, dateList, fields, offday)
    delta_1 = dfDelta(closeData, 1)
    max_1 = dfLarger(delta_1, 0)
    sma_1 = tsSma(max_1, 12, 1)
    sma_2 = tsSma(dfABS(delta_1), 12, 1) + 0.001
    temp_1 = sma_1 / sma_2 * 100
    temp_2 = dfMin(temp_1, 12)
    temp_11 = temp_1 - temp_2
    max_2 = dfMax(temp_1, 12)
    temp_22 = max_2 - temp_2 + 0.001
    result = temp_11 / temp_22

    return result.T[dateList]


def alpha_163(stockList, dateList):
    # RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    fields = 'close, high, volume, vwap, p_change'
    offday = -20

    closeData, highData, volData, vwapData, retData = generateDataFrame(stockList, dateList, fields, offday)

    result = csRank((-1 * retData) * dfMean(volData, 20) * vwapData * (highData - closeData))

    return result.T[dateList]


def alpha_164(stockList, dateList):
    # SMA((((CLOSE > DELAY(CLOSE, 1))?1 / (CLOSE - DELAY(CLOSE, 1)):1)-MIN(((CLOSE > DELAY(CLOSE, 1))?1 / (CLOSE - D
    # ELAY(CLOSE, 1)):1), 12)) / (HIGH - LOW) * 100, 13, 2)
    fields = 'close, high, low'
    offday = -26

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = closeData > dfDelay(closeData, 1)
    delta_1 = dfDelta(closeData, 1) + 0.001
    temp_1 = dfTripleOperation(condition_1, 1.0 / delta_1, 1)
    temp_2 = dfMin(temp_1, 12)
    temp_11 = temp_1 - temp_2
    result = tsSma(temp_11 / (highData - lowData + 0.001) * 100, 13, 2)

    return result.T[dateList]


def alpha_165(stockList, dateList):
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    fields = 'close'
    offday = -96

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp = closeData - dfMean(closeData, 48)
    max_1 = temp.rolling(48).apply(lambda x: cumList(x).max())
    min_1 = temp.rolling(48).apply(lambda x: cumList(x).min())
    result = (max_1 - min_1) / (dfStd(closeData, 48) + 0.001)

    return result.T[dateList]


def alpha_166(stockList, dateList):
    # -20* （ 20-1 ）^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)
    # /((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    fields = 'close'
    offday = -41

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = closeData / (dfDelay(closeData, 1) + 0.001)
    sum_1 = dfSum(temp_1 - 1 - dfMean(temp_1 - 1, 20), 20)
    sum_2 = dfSum(temp_1 ** 2, 20) + 0.001
    result = -20 * (20 - 1) ** 1.5 * sum_1 / ((20 - 1) * (20 - 2) * sum_2 ** 1.5)

    return result.T[dateList]


def alpha_167(stockList, dateList):
    # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    fields = 'close'
    offday = -13

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delta_1 = dfDelta(closeData, 1)
    condition_1 = delta_1 > 0
    result = dfSum(dfTripleOperation(condition_1, delta_1, 0), 12)

    return result.T[dateList]


def alpha_168(stockList, dateList):
    # (-1*VOLUME/MEAN(VOLUME,20))
    fields = 'volume'
    offday = -20

    volData = generateDataFrame(stockList, dateList, fields, offday)

    result = -1 * volData / (dfMean(volData, 20) + 1)

    return result.T[dateList]


def alpha_169(stockList, dateList):
    # SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
    fields = 'close'
    offday = -47

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    delay_1 = dfDelay(tsSma(dfDelta(closeData, 1), 9, 1), 1)
    result = tsSma(dfMean(delay_1, 12) - dfMean(delay_1, 26), 10, 1)

    return result.T[dateList]


def alpha_170(stockList, dateList):
    # ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /
    # 5))) - RANK((VWAP - DELAY(VWAP, 5))))
    fields = 'close, high, volume, vwap'
    offday = -20

    closeData, highData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(1.0 / (closeData + 0.001))
    temp_1 = rank_1 * volData / (dfMean(volData, 20) + 1)
    temp_2 = highData * csRank(highData - closeData)
    temp_3 = dfSum(highData, 5) / 5.0 + 0.001
    temp_11 = temp_1 * temp_2 / temp_3
    result = temp_11 - csRank(dfDelta(vwapData, 5))

    return result.T[dateList]


def alpha_171(stockList, dateList):
    # ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    fields = 'open,close, high, low'
    offday = 0

    openData, closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = (lowData - closeData) * (openData ** 5)
    temp_2 = (closeData - highData) * (closeData ** 5) + 0.001
    result = -1 * temp_1 / temp_2

    return result.T[dateList]


def alpha_172(stockList, dateList):
    # MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &
    # HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &
    # HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)

    fields = 'close ,high ,low'
    offday = -21

    # 创建dataframe数据
    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    LD = dfDelay(lowData, 1) - lowData
    HD = dfDelta(highData, 1)
    TR = dfLarger(dfLarger(highData - lowData, dfABS(highData - dfDelay(closeData, 1))),
                  dfABS(lowData - dfDelay(closeData, 1)))

    condition_1 = dfAND(LD > 0, LD > HD)
    sum_1 = dfSum(dfTripleOperation(condition_1, LD, 0), 14)
    condition_2 = dfAND(HD > 0, HD > LD)
    sum_2 = dfSum(dfTripleOperation(condition_2, HD, 0), 14)
    sum_3 = dfSum(TR, 14) + 0.001
    abs_1 = dfABS(sum_1 / sum_3 * 100 - sum_2 / sum_3 * 100)
    temp_1 = sum_1 / sum_3 * 100 + sum_2 / sum_3 * 100 + 0.001
    result = dfMean(abs_1 / temp_1 * 100, 6)

    return result.T[dateList]


def alpha_173(stockList, dateList):
    # 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(closeData, 13, 2)
    sma_2 = tsSma(sma_1, 13, 2)
    sma_3 = tsSma(tsSma(tsSma(dfLog(closeData), 13, 2), 13, 2), 13, 2)
    result = 3 * sma_1 - 2 * sma_2 + sma_3

    return result.T[dateList]


def alpha_174(stockList, dateList):
    # SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = closeData > dfDelay(closeData, 1)
    result = tsSma(dfTripleOperation(condition_1, dfStd(closeData, 20), 0), 20, 1)
    
    return result.T[dateList]


def alpha_175(stockList, dateList):
    # MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    fields = 'close, high, low'
    offday = -7

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    max_1 = dfLarger(highData - lowData, dfABS(dfDelay(closeData, 1) - highData))
    max_2 = dfLarger(max_1, dfABS(dfDelay(closeData, 1) - lowData))
    result = dfMean(max_2, 6)

    return result.T[dateList]


def alpha_176(stockList, dateList):
    # CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    fields = 'close, high, low, volume'
    offday = -18

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = closeData - dfMin(lowData, 12)
    temp_2 = dfMax(highData, 12) - dfMin(lowData, 12)
    result = rollCorr(csRank(temp_1 / temp_2), csRank(volData), 6)

    return result.T[dateList]


def alpha_177(stockList, dateList):
    # ((20-HIGHDAY(HIGH,20))/20)*100
    fields = 'high'
    offday = -20

    highData = generateDataFrame(stockList, dateList, fields, offday)

    result = (20 - highDay(highData, 20)) / 20.0 * 100
    return result.T[dateList]


def alpha_178(stockList, dateList):
    # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
    fields = 'close , volume'
    offday = -1

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfDelta(closeData, 1) / (dfDelay(closeData, 1) + 0.001) * volData

    return result.T[dateList]


def alpha_179(stockList, dateList):
    # (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    fields = 'low, volume, vwap'
    offday = -62

    lowData, volData, vwapData = generateDataFrame(stockList, dateList, fields, offday)

    rank_1 = csRank(rollCorr(vwapData, volData, 4))
    rank_2 = csRank(rollCorr(csRank(lowData), csRank(dfMean(volData, 50)), 12))
    result = rank_1 * rank_2

    return result.T[dateList]


def alpha_180(stockList, dateList):
    # ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))
    fields = 'close, volume'
    offday = -67

    closeData, volData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = dfMean(volData, 20) < volData
    temp_1 = -1 * tsRank(dfABS(dfDelta(closeData, 7)), 60)
    sign_1 = dfSign(dfDelta(closeData, 7))
    result = dfTripleOperation(condition_1, temp_1 * sign_1, -1 * volData)

    return result.T[dateList]


def alpha_181(stockList, dateList):
    # SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(B
    # ANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2)
    fields = 'close'
    offday = -42

    closeData = generateDataFrame(stockList, dateList, fields, offday)
    marketClose = closeData.mean(axis=1)
    delay_1 = dfDelay(closeData, 1) + 0.001
    temp_1 = closeData / delay_1 - 1
    mean_1 = dfMean(temp_1, 20)
    temp_2 = (marketClose- dfMean(marketClose, 20)) ** 2
    sum_1 = dfSum(np.subtract(temp_1 - mean_1, temp_2.to_frame()), 20)
    sum_2 = dfSum((marketClose- dfMean(marketClose, 20)) ** 2, 20) + 0.001
    result = 100 * (np.divide(sum_1, sum_2.to_frame()) + 1)

    return result.T[dateList]


def alpha_182(stockList, dateList):
    # COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN &
    # BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    fields = 'open, close'
    offday = -20

    openData, closeData = generateDataFrame(stockList, dateList, fields, offday)
    marketOpen, marketClose = openData.mean(axis=1), closeData.mean(axis=1)

    condition_1 = dfAND(closeData > openData, (marketClose > marketOpen).to_frame())
    condition_2 = dfAND(closeData < openData, (marketClose > marketOpen).to_frame())
    condition_12 = dfOR(condition_1, condition_2)
    result = dfCount(condition_12, 20) / 20.0

    return result.T[dateList]


def alpha_183(stockList, dateList):
    # MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    fields = 'close'
    offday = -48

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp = closeData - dfMean(closeData, 24)
    max_1 = temp.rolling(24).apply(lambda x: cumList(x).max())
    min_1 = temp.rolling(24).apply(lambda x: cumList(x).min())
    result = (max_1 - min_1) / (dfStd(closeData, 24) + 0.001)

    return result.T[dateList]


def alpha_184(stockList, dateList):
    # (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    fields = 'open, close'
    offday = -201

    openData, closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = csRank(rollCorr(dfDelay(openData - closeData, 1), closeData, 200)) + csRank(openData - closeData)

    return result.T[dateList]


def alpha_185(stockList, dateList):
    # RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
    fields = 'open, close'
    offday = 0

    openData, closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = csRank(-1 * (1 - (openData / (closeData + 0.001)) ** 2))

    return result.T[dateList]


def alpha_186(stockList, dateList):
    # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &
	# HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &
	# HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 &
	# LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 &
	# LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
    fields = 'close, high, low'
    offday = -27

    closeData, highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    LD = dfDelay(lowData, 1) - lowData
    HD = dfDelta(highData, 1)
    TR = dfLarger(dfLarger(highData - lowData, dfABS(highData - dfDelay(closeData, 1))),
                  dfABS(lowData - dfDelay(closeData, 1)))

    condition_1 = dfAND(LD > 0, LD > HD)
    sum_1 = dfSum(dfTripleOperation(condition_1, LD, 0), 14)
    condition_2 = dfAND(HD > 0, HD > LD)
    sum_2 = dfSum(dfTripleOperation(condition_2, HD, 0), 14)
    sum_3 = dfSum(TR, 14) + 0.001
    abs_1 = dfABS(sum_1 / sum_3 * 100 - sum_2 / sum_3 * 100)
    temp_1 = sum_1 / sum_3 * 100 + sum_2 / sum_3 * 100 + 0.001
    temp_11 = dfMean(abs_1 / temp_1 * 100, 6)
    temp_22 = dfDelay(temp_11, 6)
    result = (temp_11 + temp_22) / 2.0

    return result.T[dateList]


def alpha_187(stockList, dateList):
    # SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    fields = 'open, high'
    offday = -21

    openData, highData = generateDataFrame(stockList, dateList, fields, offday)

    condition_1 = openData <= dfDelay(openData, 1)
    result = dfSum(dfTripleOperation(condition_1, 0, dfLarger(highData - 
    	openData, openData - dfDelay(openData, 1))), 20)

    return result.T[dateList]


def alpha_188(stockList, dateList):
    # ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    fields = 'high, low'
    offday = -12

    highData, lowData = generateDataFrame(stockList, dateList, fields, offday)

    sma_1 = tsSma(highData - lowData, 11, 2) + 0.001
    result = (highData - lowData - sma_1) / sma_1 * 100

    return result.T[dateList]


def alpha_189(stockList, dateList):
    # MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    fields = 'close'
    offday = -12

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    result = dfMean(dfABS(closeData - dfMean(closeData, 6)), 6)

    return result.T[dateList]


def alpha_190(stockList, dateList):
    # LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(C
	# LOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-
	# 1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOS
	# E)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
    fields = 'close'
    offday = -40

    closeData = generateDataFrame(stockList, dateList, fields, offday)

    temp_1 = closeData / (dfDelay(closeData, 19) + 0.001) - 1
    temp_2 = (closeData / dfDelay(closeData, 19)) ** (1 / 20) - 1
    condition_1 = temp_1 > temp_2
    count_1 = dfCount(condition_1, 20)
    temp_11 = count_1 - 1
    sumif_1 = dfSumif((temp_1 - temp_2) ** 2, 20, temp_1 < temp_2)
    temp_22 = dfCount(temp_1 < temp_2, 20)
    sumif_2 = dfSumif((temp_1 - temp_2) ** 2, 20, temp_1 > temp_2)
    result = dfLog((temp_11 * sumif_1 + 1) / (temp_22 * sumif_2 + 0.001))

    return result.T[dateList]


def alpha_191(stockList, dateList):
    # ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    fields = 'close, high, low, volume'
    offday = -25

    closeData, highData, lowData, volData = generateDataFrame(stockList, dateList, fields, offday)

    result = rollCorr(dfMean(volData, 20), lowData, 5) + (highData + lowData) / 2.0 - closeData

    return result.T[dateList]


def dfLog(df):
    if np.any(df<0):
        raise ValueError("'dfLog' encounter negtive value")
    else:
        return df.applymap(lambda x: math.log(x))


def dfAND(bool_1, bool_2):
        return np.logical_and(bool_1, bool_2)


def dfOR(bool_1, bool_2):
    return np.logical_or(bool_1, bool_2)


def dfSumif(df, rollingday, condition):
    temp = df.copy()
    temp[condition] = 0
    return temp.rolling(rollingday).sum()


def dfCount(condition, rollingday):
    def countTrue(sr):
        temp = 0
        for data in sr:
            if data == 1.0:
                temp = temp + 1
        return temp

    return condition.rolling(rollingday).apply(lambda x: countTrue(x))


def dfABS(A):
    return A.abs()


def dfSmaller(A, B):
    if isinstance(A, int):
        temp = B.copy()
        temp[A < B] = A
        return temp
    elif isinstance(B, int):
        temp = A.copy()
        temp[A > B] = B
        return temp
    else:
        temp = A.copy()
        temp[A > B] = B[A > B]
        return temp


def dfLarger(A, B):
    if isinstance(A, int):
        temp = B.copy()
        temp[A > B] = A
        return temp
    elif isinstance(B, int):
        temp = A.copy()
        temp[A < B] = B
        return temp
    else:
        temp = A.copy()
        temp[A < B] = B[A < B]
        return temp


def dfTripleOperation(condition, A, B):
    temp = condition.copy()
    temp = temp.applymap(lambda x: 0 if x is True or False else 0)
    if isinstance(A, int):
        temp[condition] = A
    else:
        temp[condition] = A[condition]
    if isinstance(B, int):
        temp[-condition] = B
    else:
        temp[-condition] = B[-condition]
    return temp


def dfProd(df, rollingday):
    return df.rolling(rollingday).apply(lambda x: x.cumprod())


def cumList(sr):
    temp = sr.copy()
    temp_1 = 0
    for item in temp:
        item = item + temp_1
        temp_1 = item
    return temp


def dfSumAc(df, rollingday):
    return df.rolling(rollingday).apply(lambda x: cumList(x))


def dfSign(df):
    temp = df.copy()
    temp[df > 0] = 1
    temp[df == 0] = 0
    temp[df < 0] = -1
    return temp


def rollCov(df1, df2, n):
    dfCov = df1.rolling(n).cov(df2)
    dfStd = df1.rolling(n).std() * df2.rolling(n).std()
    dfCov = dfCov.replace([np.inf, -np.inf], 0)
    dfCov[dfStd == 0] = 0
    return dfCov


def dfStd(df, rollingday):
    return df.rolling(rollingday).std()


def dfMean(df, rollingday):
    return df.rolling(rollingday).mean()


def dfSum(df, rollingday):
    return df.rolling(rollingday).sum()


def dfMax(df, rollingday):
    return df.rolling(rollingday).max()


def dfMin(df, rollingday):
    return df.rolling(rollingday).min()


def dfDelay(df, rollingday):
    return df.shift(rollingday)


def dfDelta(df, rollingday):
    return df - df.shift(rollingday)


def dfREGBETA(df1, sr, n):
    condition_1 = isinstance(sr, np.ndarray)
    condition_2 = isinstance(sr, pd.core.series.Series)
    temp = df1.copy()
    if condition_1:
        temp.rolling(n).apply(lambda y: stats.linregress(x=sr, y=y)[0])
    if condition_2:
        for i in range(0, len(df1) - n):
            temp.iloc[i:i + n, :].apply(lambda y: stats.linregress(x=sr.iloc[i:i + n], y=y)[0])
    return temp


def dfREGRESI(df1, sr, n):
    condition_1 = isinstance(sr, np.ndarray)
    condition_2 = isinstance(sr, pd.core.series.Series)
    temp = df1.copy()
    if condition_1:
        temp.rolling(n).apply(lambda y: stats.linregress(x=sr, y=y)[1])
    if condition_2:
        for i in range(0, len(df1) - n):
            temp.iloc[i:i + n, :].apply(lambda y: stats.linregress(x=sr.iloc[i:i + n], y=y)[1])
    return temp


def highDay(sr, n):
    return sr.rolling(n).apply(lambda x: x[::-1].argmax())


def lowDay(sr, n):
    return sr.rolling(n).apply(lambda x: x[::-1].argmin())


# Cross-section Rank, df:DataFrame(Clomns MUST be Codes, Rows MUST be Dates)
def csRank(df):
    return df.apply(lambda x: x.rank(pct=True), axis=1)


# Time-series Rank, sr:Seires, n:window=n
def tsRank(sr, n):
    return sr.rolling(n).apply(lambda x: (sorted(x).index(x[-1]) + 1)/ float(n))

# Time-series SMA, sr:Seires, n:window=n, m:weight(m < n)
def tsSma(sr, n, m):
    def weightedAVG(y, a):
        return ((n - m) * y + m * a) / n

    return sr.rolling(n).apply(lambda x: reduce(weightedAVG, x))


# rolling correlation
def rollCorr(df1, df2, n):
    dfCorr = df1.rolling(n).corr(df2)
    dfStd = df1.rolling(n).std() * df2.rolling(n).std()
    dfCorr = dfCorr.replace([np.inf, -np.inf], 0)
    dfCorr[dfStd == 0] = 0
    return dfCorr


# DECAYLINEAR
def decayLinear(sr, d):
    y = list(range(1, d + 1))
    return sr.rolling(d).apply(lambda x: sum(np.multiply(x, y)) / sum(y))


# WMA
def tsWma(sr, n):
    y = np.power(0.9, list(range(1, n + 1))[::-1])
    return sr.rolling(n).apply(lambda x: sum(np.multiply(x, y)) / sum(y))


def cumReturn(df, d, axis=0):
    df = df.rolling(d, axis=axis).apply(lambda x: 100 * (np.add(x / 100.0, 
                   1).prod() - 1)).shift(1-d, axis=axis)
    
    return df


def dfFilter(df, condition):
    temp = df.copy()
    # temp = temp[condition]
    temp = temp[condition].dropna(axis = 0,how = 'all')
    temp = temp.replace([np.inf, -np.inf, np.nan], 0)
    return temp

#####################################################
def file2List(file):
    fp = open(file, 'r')
    outList = []
    for line in fp.readlines():
        outList.append(line.split('\n')[0])
    fp.close()
    return outList


def tDateOffset(inDateStr, offSetDays, timeSerialFile=timeSerialFile):
    timeSerialList = open(timeSerialFile).read().split('\n')
    inDateIdx = timeSerialList.index(inDateStr)
    outDateIdx = inDateIdx + np.sign(offSetDays) * (abs(offSetDays))
    outDate = timeSerialList[outDateIdx]
    if offSetDays > 0:
        dateList = timeSerialList[inDateIdx:outDateIdx + 1]
    else:
        dateList = timeSerialList[outDateIdx:inDateIdx + 1]
    return outDate, dateList

def tDaysOffset(inDateList, offSetDays):
    if offSetDays > 0:
        expandDays = tDateOffset(inDateList[-1], offSetDays)[1]
        return inDateList + expandDays[0:]
    else:
        expandDays = tDateOffset(inDateList[0], offSetDays)[1]
        return expandDays[:-1] + inDateList

def getDateIntvlList(beginDateStr, endDateStr):
    listFull = open(timeSerialFile).read().split('\n')
    beginIdx = listFull.index(beginDateStr)
    endIdx = listFull.index(endDateStr)
    outList = listFull[beginIdx:endIdx+1]
    return outList

def getDateListFromFile(inputFile):    
    fp = open(inputFile, 'r')
    dateListInit = fp.readlines()
    fp.close()
    outList = [dateStr[:-1] for dateStr in dateListInit]
    return outList

def findPreTdate(dateStr):
    tDateList = file2List(timeSerialFile)
    if dateStr not in tDateList:
        diffList = np.array(tDateIntList) - int(dateStr)
        idxArray = np.where(diffList<0)
        idx = idxArray[0][-1]
        outDateStr = tDateList[idx]
    else:
        outDateStr = dateStr
    return outDateStr

def findAfterTdate(dateStr):
    tDateList = file2List(timeSerialFile)
    if dateStr not in tDateList:
        outDateStr = tDateOffset(findPreTdate(dateStr), 1, timeSerialFile)[0]
    else:
        outDateStr = dateStr
    return outDateStr

def getTradeDate(start, end):
    tDateList = file2List(timeSerialFile)
    dateList = [x for x in tDateList if (x>=start) and (x<=end)]
    return dateList 

def tMonthOffset(inMonthStr, offSetMonths):
    timeSerialList = open(monthCalendarFile).read( ).split('\n')  
    inMonthIdx = timeSerialList.index(inMonthStr)
    outMonthIdx = inMonthIdx + np.sign(offSetMonths) * (abs(offSetMonths) -1)
    outMonth  = timeSerialList[outMonthIdx]
    if offSetMonths >0:
        monthList = timeSerialList[inMonthIdx:outMonthIdx+1]
    else:
        monthList = timeSerialList[outMonthIdx:inMonthIdx+1]    
    return outMonth, monthList

def getMonthTdate(monthStr):
    dateList = file2List(tradeCalendarFile)
    outList = []
    for dateStr in dateList:
        if dateStr[:6]==monthStr:
            outList.append(dateStr)    
    return outList

#############################################################

def data_pre(stockList, dateList, path=minute_file_path):

    df = pd.DataFrame()  
    if len(dateList) > 1:
        for f in stockList:
            fn = f[-2:] + f[:6] + '.feather'
            df_0 = ft.read_dataframe(path + fn, nthreads=100).set_index('date')
            df_1 = df_0.loc[dateList, :]
            df_1['code'] = f
            df = df.append(df_1)
    else:
        for f in stockList:
            fn = f[-2:] + f[:6] + '.feather'
            df_0 = ft.read_dataframe(path + fn, nthreads=100).set_index('date')
            df_1 = df_0.loc[dateList[0], :]
            df_1 = df_1.to_frame().T
            df_1['code'] = f
            df = df.append(df_1)
    df['volume'] = df.amount / df.vwap
    df['p_change'] = (df.close / df.preClose - 1) * 100
    
    return df

def generateDataFrame(stockList, dateList, fields, offday):
    fullDate = tDaysOffset(dateList, offday)
    tmp = data_pre(stockList, fullDate)
    tmp = tmp[tmp.amount >= 1]
    tmp = tmp.pivot(columns='code')
    field = [item.strip() for item in fields.split(',')]
    result = []
    for f in field:
        result.append(tmp[f])
    if len(result) == 1:
        return result[0]
    else:
        return result
 
