#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:51:39 2017

@author: lishiwang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from backtest import backtest
from barra_base import barra_base
from db_engine import db_engine
from database import database

class msci_stocks():
    def __init__(self, *, startdate):
        stock_list = pd.read_excel('msci list.xlsx', sheetname=1)
        self.stock_isin = stock_list['ISIN']
        self.db = database(start_date=startdate)
        self.db.initialize_jydb()
        self.db.initialize_sq()

    def get_stock_code(self):
        # 创建包含isin代码的sql string
        isin_sql_str = ''
        for i in self.stock_isin.iteritems():
            if type(i[1]) != str and np.isnan(i[1]):
                continue
            isin_sql_str += "'" + str(i[1]) + "'"
            if i[0] <self.stock_isin.size - 1:
                isin_sql_str += ", "
        sql_query = "select distinct SecuCode from SecuMain where ISIN in (" + isin_sql_str + ") order by SecuCode"
        self.msci_stockcode = self.db.jydb_engine.get_original_data(sql_query)
        self.msci_stockcode = self.msci_stockcode.iloc[:, 0]
        pass

    def get_price_data(self):
        self.db.get_trading_days()
        self.db.get_labels()
        self.db.get_ClosePrice_adj()
        self.db.get_index_price()
        # 获取收益数据
        closeprice_adj = self.db.data.stock_price.ix['ClosePrice_adj']
        self.stock_return = np.log(closeprice_adj/closeprice_adj.shift(1)).fillna(0)
        self.benchmark_return = np.log(self.db.data.benchmark_price.div(self.db.data.benchmark_price.shift(1).reindex(
                major_axis=self.db.data.stock_price.major_axis))).fillna(0)

    def get_msci_stocks_stats(self):
        # 等权持有的收益
        self.port_value_ew = np.exp(self.stock_return.reindex(columns=self.msci_stockcode).cumsum()).mean(1)
        self.benchmark_value = np.exp(self.benchmark_return.cumsum(1))
#        self.port_value_ew = self.stock_return.reindex(columns=self.msci_stockcode).cumsum().mean(1)
#        self.benchmark_value = self.benchmark_return.cumsum(1)
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(self.port_value_ew, label='stocks')
        ax.plot(self.benchmark_value.ix['ClosePrice_zz500', :, 0], label='zz500')
        ax.plot(self.benchmark_value.ix['ClosePrice_hs300', :, 0], label='hs300')
        ax.plot(self.benchmark_value.ix['ClosePrice_zz800', :, 0], label='zz800')
        ax.plot(self.benchmark_value.ix['ClosePrice_sz50', :, 0], label='sz50')
        ax.legend(loc='best')
        
if __name__ == '__main__':
    msci_s = msci_stocks(startdate='2017-03-02')
    msci_s.get_stock_code()
    msci_s.get_price_data()
    msci_s.get_msci_stocks_stats()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

