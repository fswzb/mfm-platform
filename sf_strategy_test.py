#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:17:06 2017

@author: lishiwang
"""


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('AGG')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
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
                           stock_pools=['all', 'hs300', 'zz500', 'zz800'], bkt_start='default', bkt_end='default',
                           do_bb_pure_factor=False, do_pa=False, select_method=0, do_active_pa=False):
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
        bb_obj.just_get_sytle_factor()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    # 根据股票池进行循环
    for stock_pool in stock_pools:
        # 建立单因子测试对象
        curr_sf = single_factor_strategy()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，尽管这里并不是必要的
        curr_sf.single_factor_test(factor=factor, direction=direction, bkt_obj=copy.deepcopy(bkt_obj),
                                   bb_obj=copy.deepcopy(bb_obj), discard_factor=discard_factor,
                                   bkt_start=bkt_start, bkt_end=bkt_end,
                                   stock_pool=stock_pool, do_pa=do_pa, do_bb_pure_factor=do_bb_pure_factor,
                                   select_method=select_method, do_active_pa=do_active_pa)

# 根据多个股票池进行一次完整的单因子测试, 多进程版
def sf_test_multiple_pools_parallel(factor, *, direction='+', bb_obj='Empty', discard_factor=[],
                                    stock_pools=['all', 'hs300', 'zz500', 'zz800'], bkt_start='default',
                                    bkt_end='default', do_bb_pure_factor=False, do_pa=False,
                                    select_method=0, do_active_pa=False):
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
        bb_obj.just_get_sytle_factor()
    # 外部传入的bb对象，要检测其股票池是否为all，如果不是all，则输出警告，因为可能丢失了数据
    elif bb_obj.bb_data.stock_pool != 'all':
        print('The stockpool of the barra_base obj from outside is NOT "all", be aware of possibile'
              'data loss due to this situation!\n')

    def single_task(stock_pool):
        curr_sf = single_factor_strategy()

        # 进行当前股票池下的单因子测试
        # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
        # 对bkt obj做了同样的处理，这是因为尽管bkt obj不会被改变，但是多进程同时操作可能出现潜在的问题
        curr_sf.single_factor_test(factor=factor, stock_pool=stock_pool, direction=direction,
                                   bkt_obj=copy.deepcopy(bkt_obj), bb_obj=copy.deepcopy(bb_obj),
                                   discard_factor=discard_factor, bkt_start=bkt_start, bkt_end=bkt_end,
                                   do_pa=do_pa, do_bb_pure_factor=do_bb_pure_factor,
                                   select_method=select_method, do_active_pa=do_active_pa)

    from multiprocessing import Process
    # 根据股票池进行循环
    for stock_pool in stock_pools:
        Process(target=single_task, args=(stock_pool,)).start()


# 进行单因子测试

# 测试eps_fy1, eps_fy2的varaition coeffcient
eps_fy = data.read_data(['EPS_fy1', 'EPS_fy2'])
eps_fy1 = eps_fy['EPS_fy1']
eps_fy2 = eps_fy['EPS_fy2']

eps_vc = 0.5*eps_fy1.rolling(252).std()/eps_fy1.rolling(252).mean() + \
        0.5*eps_fy1.rolling(252).std()/eps_fy1.rolling(252).mean()

# # 测试wq101中的因子
# wq_data = data.read_data(['ClosePrice_adj', 'OpenPrice_adj', 'Volume'],
#                          ['ClosePrice_adj', 'OpenPrice_adj', 'Volume'], shift=True)
# # 因子3
# f3_open_rank = wq_data.ix['OpenPrice_adj'].rank(1)
# f3_vol_rank = wq_data.ix['Volume'].rank(1)
# f3_open_rank_rolling = f3_open_rank.rolling(10)
# f3_vol_rank_rolling = f3_vol_rank.rolling(10)
# wq_f3 = -f3_open_rank_rolling.corr(f3_vol_rank_rolling)
#
# #因子2
# f2_log_vol = np.log(wq_data.ix['Volume'])
# f2_vol_rank_rolling = (f2_log_vol-f2_log_vol.shift(2)).rank(1).rolling(6)
# f2_ret_rank_rolling = (wq_data.ix['ClosePrice_adj']/wq_data.ix['OpenPrice_adj']-1).rank(1).rolling(6)
# wq_f2 = -f2_vol_rank_rolling.corr(f2_ret_rank_rolling)


#sf_test_multiple_pools(factor=wq_f3, direction='+', bkt_start=pd.Timestamp('2009-03-03'),
#                          bkt_end=pd.Timestamp('2017-03-30'), stock_pools=['hs300'],
#                          do_bb_pure_factor=False, do_pa=False)

sf_test_multiple_pools_parallel(factor=eps_vc, direction='-', bkt_start=pd.Timestamp('2009-03-03'),
                           bkt_end=pd.Timestamp('2017-03-30'), stock_pools=['zz500', 'zz800'],
                           do_bb_pure_factor=True, do_pa=True, select_method=0, do_active_pa=False)


































































