#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepdata_sp500.py

Purpose:
    Prepare the data on SP500

Version:
    1       First start, outline for students to start with

Date:
    2025/9/22

Author:
    ???
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import glob
import os

###########################################################
### df= PrepSPY(dtArg)
def PrepSPY(dtArg):
    sGlob = 'data/Price*_all_*_i0.xlsx'

    asF = np.sort(glob.glob(sGlob))

    lsSymbols = dtArg['symbols'].split()



    dfList = []
    for f in asF:
        dfTemp = pd.read_excel(f, header=None)
        dfTemp.columns = dfTemp.iloc[0].str.strip()  # remove spaces from headers
        dfTemp = dfTemp.drop(0)

        # Ensure all requested symbols exist in this dfTemp
        colsToKeep = ['Date'] + [c for c in lsSymbols if c in dfTemp.columns]
        dfTemp = dfTemp[colsToKeep]

        # Add missing symbols as NaN
        for sym in lsSymbols:
            if sym not in dfTemp.columns:
                dfTemp[sym] = np.nan

        # Reorder columns: Date first, then symbols in lsSymbols order
        dfTemp = dfTemp[['Date'] + lsSymbols]

        dfList.append(dfTemp)


    if dfList:
        df = pd.concat(dfList).drop_duplicates().sort_values('Date').reset_index(drop=True)
    else:
        df = pd.DataFrame()
    
    print('#Data points: ' + str(len(df)))
    df.dropna(inplace=True)
    print('#Data points after removing nans:' + str(len(df)))

    return df

###########################################################
### main
def main():
    # Magic numbers
    dtArg= {
        'symbols': 'SPX5.L SPY5l.AQX SPY5.MIL',        # Change list of symbols to the symbols
        'group': 'g2'
    }

    # Initialisation
    # Initialise(dtArg)

    # Estimation
    df= PrepSPY(dtArg)
    print(df.head())

    # Output
    os.makedirs("output", exist_ok=True)
    sOut= f'output/sp_{dtArg["group"]}.csv'
    df.to_csv(sOut)

    print (f'See {df.shape} observations in {sOut}')
    print ('Beginning of dataset:')
    print (df.head())

###########################################################
### start main
if __name__ == "__main__":
    main()
