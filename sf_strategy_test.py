#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:17:06 2017

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

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from backtest import backtest
from barra_base import barra_base
from single_factor_strategy import single_factor_strategy

# 根据多个股票池进行一次完整的单因子测试
def sf_test_multiple_pools(factor, *, direction='+', bb_obj='Empty', discard_factor=[],
                           stock_pools=['all', 'hs300', 'zz500', 'zz800'], bkt_start='default', bkt_end='default'):
    # 如果传入的是str，则读取同名文件，如果是dataframe，则直接传入因子
    # 注意：这里的因子数据并不储存到self.strategy_data.factor中，因为循环股票池会丢失数据
    # 这里实际上是希望每次循环都有一个新的single factor strategy对象，
    # 而这个函数的目的则是每次循环不用再重新建立这个对象
    if type(factor) == str:
        factor_data = data.read_data([factor], [factor], shift=True)
        factor = factor_data[factor]

    # 初始化一个持仓对象，以用来初始化backtest对象，索引以factor为标准
    temp_position = position(factor)
    # 先要初始化bkt对象
    bkt_obj = backtest(temp_position, bkt_start=bkt_start, bkt_end=bkt_end)
    # 建立bb对象，否则之后每次循环都要建立一次新的bb对象
    if bb_obj == 'Empty':
        bb_obj = barra_base()
        bb_obj.construct_barra_base()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    # 根据股票池进行循环
    for stock_pool in stock_pools:
        # 建立单因子测试对象
        curr_sf = single_factor_strategy()
        # 进行当前股票池下的单因子测试
        curr_sf.single_factor_test(factor=factor, direction=direction, bkt_obj=bkt_obj, bb_obj=bb_obj,
                                   discard_factor=discard_factor, bkt_start=bkt_start, bkt_end=bkt_end,
                                   stock_pool=stock_pool)

## 建立单因子策略对象
#sf = single_factor_strategy()

#sf.single_factor_test(factor='FreeMarketValue', direction='-', bkt_start=pd.Timestamp('2009-03-03'),
#                      bkt_end=pd.Timestamp('2017-03-30'))

sf_test_multiple_pools(factor='FreeMarketValue', direction='-', bkt_start=pd.Timestamp('2009-03-03'),
                          bkt_end=pd.Timestamp('2017-03-30'), stock_pools=['all', 'hs300','zz500','zz800'])