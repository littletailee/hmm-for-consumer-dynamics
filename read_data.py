# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:49:03 2018

@author: jwzhang1996
"""

import pandas as pd
import numpy as np


def read_data():
    dat = pd.read_csv('act2000b.dat', sep='	')
    dat['volunteer'] = dat['volunteer'].apply(lambda x: 0 if x == '.' else 1)
    int_loc = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
    dat.iloc[:, int_loc] = dat.iloc[:, int_loc].apply(
        lambda x: x.apply(int, 1))
    dat.iloc[:, [2, 3]] = dat.iloc[:, [2, 3]].apply(
        lambda x: x.apply(float, 1))
    dat['time_after_grad'] = dat['Year'] - dat['grad year']
    dat['reunion_year'] = np.zeros(len(dat.iloc[:, 1]))
    row = dat[dat['reuniun'] == 1].index
    dat['reunion_year'][row] = dat['Year'][row]

    grouped = dat.groupby('ID')
    c = grouped[['Year']].agg(['count'])

    IDs = c[c.iloc[:, 0] == 23].index
    df = dat[dat['ID'].isin(IDs)]
    names = list(df)
    sequences = df.groupby(['ID'])
    data = sequences[names].apply(lambda x: x.values.tolist()).tolist()

    del grouped, c, df, sequences, IDs, int_loc, row

    # ----------------------------------here we go-----------------------------------------------------------

    # all data in tensor_all: array(#ID 741* #year 23* #variable 17)
    # variables: 0'ID', 1'Year',2'amount', 3'lag_amount', 4'Gift (yes/no)', 5'Lag_gift', 6'Years in data',
    #           7'Gift years', 8'SAA member',  9'Spouse alum', 10'number of degrees', 11'grad year', 12'reuniun',
    #           13'award', 14'volunteer', 15'Event', 16'time_after_grad'，17'reunion_year'
    tensor_all = np.array(data)

    #a_it: array(741, 23, 3)
    # variables: 0'reuniun', 1'volunteer', 2'Year'
    a_it = tensor_all[:, :, [12, 14, 1]]

    #x_it: array(741, 23, 3)
    # variables: 0'reunion year', 1'time after graduation'
    # fail to find reunion year-------------------
    x_it = tensor_all[:, :, [16]]
    # potential one:
    #x_it = tensor_all[:,:,[16,17]]

    #y_it: array(741, 23, 1)
    # variable: 0'Gift(yes/no)'
    y_it = tensor_all[:, :, [4]]

    # z_it: array(741, 23, 3) (personal information, static)
    # variable: 0'SAA member', 1'Spouse alum', 2'number of degrees'
    z_it = tensor_all[:, :, [8, 9, 10]]

    return a_it, x_it, y_it, z_it
