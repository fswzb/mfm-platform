#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:24:49 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from strategy_data import strategy_data
from backtest_data import backtest_data
from position import position
from strategy import strategy
from single_factor_strategy import single_factor_strategy
from backtest import backtest


# 定义按因子分位数选股的函数，将不同分位数收益率画到一张图上，同时还会画long-short的图
# value=1为画净值曲线图，value=2为画对数收益率图，holding=0为等权，=1为市值加权
def plot_qgroup(strategy_class, bkt, no_of_groups, *, direction = '+', value = 1, holding = 0):
    # 默认画净值曲线图
    if value == 1:
        # 先初始化图片
        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Net Account Value')
        ax1.set_title('Net Account Value Comparison of Different Quantile Groups of The Factor')
    
        # 开始循环选股、画图
        for group in range(no_of_groups):
            # 选股
            strategy_class.reset_position()
            strategy_class.select_qgroup(no_of_groups, group+1, direction = direction, holding = holding)
        
            # 回测
            bkt.reset_bkt_position(strategy_class.position)
            bkt.execute_backtest()
            bkt.initialize_performance()
        
            # 画图，注意，这里画净值曲线图，差异很小时，净值曲线图的差异更明显
            plt.plot(bkt.bkt_performance.net_account_value, label = 'Group %s' % str(group+1))
        
            # 储存第一组和最后一组以画long-short收益图
            if group == 0:
                long_series = bkt.bkt_performance.net_account_value
            elif group == no_of_groups-1:
                short_series = bkt.bkt_performance.net_account_value
        
        ax1.legend(loc = 'best')
 
        # 画long-short的图
        f2 = plt.figure()
        ax2 = f2.add_subplot(1,1,1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Net Account Value')
        ax2.set_title('Net Account Value of Long-Short Portfolio of The Factor')
        plt.plot(long_series - short_series)
        
    elif value == 2:
        # 先初始化图片
        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Log Return (%)')
        ax1.set_title('Cumulative Log Return Comparison of Different Quantile Groups of The Factor')
    
        # 开始循环选股、画图
        for group in range(no_of_groups):
            # 选股
            strategy_class.reset_position()
            strategy_class.select_qgroup(no_of_groups, group+1, direction = direction, holding = holding)
        
            # 回测
            bkt.reset_bkt_position(strategy_class.position)
            bkt.execute_backtest()
            bkt.initialize_performance()
        
            # 画图，注意，这里画累积对数收益图，当差异很大时，累积对数收益图看起来更容易
            plt.plot(bkt.bkt_performance.cum_log_return*100, label = 'Group %s' % str(group+1))
        
            # 储存第一组和最后一组以画long-short收益图
            if group == 0:
                long_series = bkt.bkt_performance.cum_log_return
            elif group == no_of_groups-1:
                short_series = bkt.bkt_performance.cum_log_return
        
        ax1.legend(loc = 'best')
 
        # 画long-short的图
        f2 = plt.figure()
        ax2 = f2.add_subplot(1,1,1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Log Return (%)')
        ax2.set_title('Cumulative Log Return of Long-Short Portfolio of The Factor')
        plt.plot((long_series - short_series)*100)
    
    else:
        print('Please enter a valid number for argument "value"\n')
    

        
# 单因子测试主函数
def single_factor_test(factor_name, *, direction = '+', no_of_groups = 5, figure_value = 1, 
                       select_ratio = [0.8, 1], bkt_start = 'default', bkt_end = 'default', holding = 0):
    
    # 先做一个简单的收益率测试                    
    sf_strategy = single_factor_strategy()
                            
    sf_strategy.read_factor_data([factor_name],[factor_name], shift = True)
    sf_strategy.strategy_data.discard_untradable_data()
    sf_strategy.generate_holding_days()
    sf_strategy.initialize_position(sf_strategy.strategy_data.factor.ix[0, sf_strategy.holding_days, :])
                            
    sf_strategy.select_stocks(select_ratio = select_ratio, direction = direction, holding = holding)
                            
    # 回测，画图
    bkt = backtest(sf_strategy.position, bkt_start = bkt_start, bkt_end = bkt_end)
    bkt.execute_backtest()
    bkt.get_performance()

    # 画因子收益走势图
    sf_strategy.get_factor_return()
    # 计算IC，画IC的走势图
    sf_strategy.get_factor_ic(direction = direction)
    
    # 画分位数和long-short图     
    plot_qgroup(sf_strategy, bkt, no_of_groups, direction = direction, value = figure_value, holding = holding)


# 测试
import time
start_time = time.time()
single_factor_test('f55', direction = '+', bkt_end = pd.Timestamp('2016-07-07'),
                   bkt_start = pd.Timestamp('2006-01-06'), holding = 1)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()




















































































