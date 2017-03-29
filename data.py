#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:00:02 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

# 数据类，所有数据均为pd.Panel, major_axis为时间，minor_axis为股票代码，items为数据名称

# 基本数据类
class data(object):
    """ This is the base class of a group of data classes, which is mostly used.
    
    stock_price (pd.Panel): price data of stocks, note the difference between stock_price
                         data and raw_data
    raw_data (pd.Panel): original data get from market or financial report, or intermediate data
                         which is used for factor calculation, note the difference between stock_price
                         data and raw_data
    benchmark_price (pd.Panel): price data of benchmarks
    if_tradable (pd.Panel): marks which indicate if stocks are enlisted, delisted, suspended, tradable, in stock pool
                            or investable.
    const_data (pd.DataFrame): const data, usually macroeconomic data, such as risk free rate or inflation rate.
    """
    
    def __init__(self):
        self.stock_price = pd.Panel()
        self.raw_data = pd.Panel()
        self.benchmark_price = pd.Panel()
        self.if_tradable = pd.Panel()
        self.const_data = pd.DataFrame()

    # 读取数据的函数
    @staticmethod
    def read_data(file_name, item_name='default', *, shift = False):
        """ Get the data from csv file.
        
        file_name: name of the file.
        item_name: name of the data in the panel.
        shift: denote that if the data read need to be shifted by 1. This is because, the strategy data we got
        will have one lag(i.e. on the start of day 2, you can not know data on day 2 but day 1 or before day 2,
        thus the decision you make on the start of day 2 is based on data before day 2.), while for backtest
        data, we don't need this lag. The default option will not condunct the lag, you can set shift to True
        to create the lag for strategy data.)
        """
        # 从文件中读取数据
        obj = {}
        for i, s in enumerate(file_name):
            temp_df = pd.read_csv(str(os.path.abspath('.'))+'/'+s+'.csv',
                                   index_col = 0, parse_dates = True, thousands=',')
            if shift:
                temp_df = temp_df.shift(1)
            if file_name=='default':
                obj[file_name[i]] = temp_df
            else:
                obj[item_name[i]] = temp_df
        obj = pd.Panel.from_dict(obj)
        return obj

    # 写数据的函数
    @staticmethod
    def write_data(written_data, *, file_name='default'):
        """ Write the data to csv file

        :param written_data: (pd.Panel) name of data to be written to csv file
        :param file_name: (list) list of strings containing names of csv files, note it has to be the same length of
        items in written_data, if it sets to default, the file name will be the name of items of the written data
        """
        if file_name == 'default':
            for cursor, item_name in enumerate(written_data.items):
                written_data.ix[cursor].to_csv(str(item_name)+'.csv', index_label='datetime', na_rep='NaN')
        else:
            for cursor, item_name in enumerate(written_data.items):
                written_data.ix[cursor].to_csv(file_name[cursor]+'.csv', index_label='datetime', na_rep='NaN')
        
    # 重新对齐索引的函数
    @staticmethod
    def align_index(standard, raw_data, *, axis = 'both'):
        """Align the index of second data to first data.
        
        standard (pd.DataFrame): data of standard index
        raw_data (pd.Panel): data to be aligned
        """
        if axis is 'both':
            aligned_data = raw_data.reindex(major_axis = standard.index, 
                                            minor_axis = standard.columns)
        elif axis is 'major':
            aligned_data = raw_data.reindex(major_axis = standard.index)
        elif axis is 'minor':
            aligned_data = raw_data.reindex(minor_axis = standard.columns)
        return aligned_data
    
    # 读取上市、退市、停牌数据，并生成可否交易的矩阵
    def generate_if_tradable(self, *, file_name = ['is_enlisted','is_delisted','is_suspended'], 
                             item_name = ['is_enlisted','is_delisted','is_suspended'], 
                             shift = False):
        # 读取上市、退市、停牌数据
        self.if_tradable = data.read_data(file_name, item_name, shift = shift)
        # 将已上市且未退市，未停牌的股票标记为可交易(if_tradable = True)
        self.if_tradable['if_tradable'] = (self.if_tradable.ix['is_enlisted', :, :] * \
                                           np.logical_not(self.if_tradable.ix['is_delisted', :, :]) * \
                                           np.logical_not(self.if_tradable.ix['is_suspended', :, :])).fillna(False).astype(np.bool)
            
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            