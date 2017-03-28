#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:32:36 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy

# 单因子表现测试

class single_factor_strategy(strategy):
    """ Single factor test strategy.
    
    foo
    """
    def __init__(self):
        strategy.__init__(self)
        self.strategy_data.generate_if_tradable(shift = True)
        # 读取市值数据以进行市值加权
        self.strategy_data.stock_price = data.read_data(['MarketValue'],['MarketValue'],shift = True)
        
    # 读取因子数据的函数
    def read_factor_data(self, file_name, factor_name, *, shift = True):
        self.strategy_data.factor = data.read_data(file_name, factor_name, shift = shift)
        
    # 选取股票，选股比例默认为最前的80%到100%，方向默认为因子越大越好，holding=1为市值加权，0为等权
    def select_stocks(self, *, select_ratio = [0.8, 1], direction = '+', holding = 0):
        # 对调仓期进行循环
        for cursor, time in self.holding_days.iteritems():
            curr_factor_data = self.strategy_data.factor.ix[0, time, :]
            # 对因子值进行排序，注意这里的秩（rank），类似于得分
            if direction is '+':
                factor_score = curr_factor_data.rank(ascending = True)
            elif direction is '-':
                factor_score = curr_factor_data.rank(ascending = False)
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
            
            # 取有效的股票数
            effective_num = curr_factor_data.dropna().size
            # 无股票可选，进行下一次循环
            if effective_num == 0:
                continue
            # 选取股票的得分范围
            lower_bound = np.floor(effective_num * select_ratio[0])
            upper_bound = np.floor(effective_num * select_ratio[1])
            # 选取股票
            selected_stocks = curr_factor_data.ix[np.logical_and(factor_score>=lower_bound, 
                                                                 factor_score<=upper_bound)].index
            # 被选取的股票都将持仓调为1
            self.position.holding_matrix.ix[time, selected_stocks] = 1
            
        # 循环结束
        # 去除不可交易的股票
        self.filter_untradable()
        # 设置为等权重
        self.position.to_percentage()
        # 如果需要市值加权，则市值加权
        if holding == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['MarketValue',
                                           self.position.holding_matrix.index, :])
    
    # 单因子的因子收益率计算和检验，用来判断因子有效性，回归收益为调仓日之间的收益率
    def get_factor_return(self):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算股票对数收益以及因子暴露
        holding_day_price = self.strategy_data.stock_price.ix['ClosePrice_adj',self.holding_days,:]
        holding_day_return = np.log(holding_day_price.div(holding_day_price.shift(1)))
        holding_day_factor = self.strategy_data.factor.ix[0, self.holding_days, :]
        holding_day_factor_expo = strategy_data.get_cap_wgt_exposure(holding_day_factor,
                                    self.strategy_data.stock_price.ix['MarketValue', self.holding_days, :])
        # 注意因子暴露要用前一期的数据
        holding_day_factor_expo = holding_day_factor_expo.shift(1)

        # 初始化因子收益序列以及估计量的t统计量序列
        factor_return_series = np.empty(self.holding_days.size)*np.nan
        t_stats_series = np.empty(self.holding_days.size)*np.nan
        self.factor_return_series = pd.Series(factor_return_series, index=self.holding_days)
        self.t_stats_series = pd.Series(t_stats_series, index=self.holding_days)

        # 进行回归，对调仓日进行循环
        for cursor, time in self.holding_days.iteritems():

            y = holding_day_return.ix[time, :]
            x = holding_day_factor_expo.ix[time, :]
            if y.isnull().all() or x.isnull().all():
                continue
            x = sm.add_constant(x)
            results = sm.OLS(y, x, missing='drop').fit()
            self.factor_return_series.ix[time] = results.params[1]
            self.t_stats_series.ix[time] = results.tvalues[1]

        # 循环结束，输出结果
        print('The average return of this factor: %f%%\n' % (self.factor_return_series.mean()*100))
        print('Note that the return of factor is not annualized but corresponding to the holding days interval\n')
        print('The average t-statistics value: %f\n' % (self.t_stats_series.mean()))
        tstats_sig_ratio = self.t_stats_series[np.abs(self.t_stats_series)>=2].size / self.t_stats_series.size
        print('Ratio of t-stats whose absolute value >= 2: %f\n' % (tstats_sig_ratio))

        # 画图
        fx = plt.figure()
        ax = fx.add_subplot(1,1,1)
        plt.plot(self.factor_return_series*100, 'b-o')
        zero_series = pd.Series(np.zeros(self.factor_return_series.shape), index = self.factor_return_series.index)
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return of The Factor (%)')
        ax.set_title('The Return Series of The Factor')

        fx = plt.figure()
        ax = fx.add_subplot(1, 1, 1)
        plt.plot(self.t_stats_series, 'b-o')
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('T-Stats of The Factor Return')
        ax.set_title('The T-Stats Series of The Factor Return')
            
    # 计算因子的IC，股票收益率为调仓日间的收益率
    def get_factor_ic(self, *, direction = '+'):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 初始化ic矩阵
        ic_series = np.empty(self.holding_days.size)*np.nan
        self.ic_series = pd.Series(ic_series, index = self.holding_days)
        # 计算股票对数收益，提取因子值，同样的，因子值要用前一期的因子值
        holding_day_price = self.strategy_data.stock_price.ix['ClosePrice_adj',self.holding_days,:]
        holding_day_return = np.log(holding_day_price.div(holding_day_price.shift(1)))
        holding_day_factor = self.strategy_data.factor.ix[0, self.holding_days, :]
        holding_day_factor = holding_day_factor.shift(1)
        # 对调仓日进行循环
        for cursor, time in self.holding_days.iteritems():
            # 计算因子值的排序
            curr_factor_data = holding_day_factor.ix[time, :]
            # 对因子值进行排序，注意这里的秩（rank），类似于得分
            if direction is '+':
                factor_score = curr_factor_data.rank(ascending = True)
            elif direction is '-':
                factor_score = curr_factor_data.rank(ascending = False)
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
            
            # 对因子实现的对数收益率进行排序，升序排列，因此同样，秩类似于得分
            return_score = holding_day_return.ix[time, :].rank(ascending = True)
            
            # 计算得分（秩）之间的线性相关系数，就是秩相关系数
            self.ic_series.ix[time] = factor_score.corr(return_score, method = 'pearson')
            
        # 循环结束
        # 输出结果
        print('The average IC of this factor: %f\n' % (self.ic_series.mean()))
        
        # 画图
        fx = plt.figure()
        ax = fx.add_subplot(1,1,1)
        plt.plot(self.ic_series, 'b-o')
        # 画一条一直为0的图，以方便观察IC的走势是否显著不为0
        zero_series = pd.Series(np.zeros(self.ic_series.shape), index = self.ic_series.index)
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('IC of The Factor')
        ax.set_title('The IC Time Series of The Factor')
        
    # 根据分位数分组选股，用来画同一因子不同分位数分组之间的收益率对比，以此判断因子的有效性
    def select_qgroup(self, no_of_groups, group, *, direction = '+', holding = 0):
        # 对调仓期进行循环
        for cursor, time in self.holding_days.iteritems():
            curr_factor_data = self.strategy_data.factor.ix[0, time, :]
            # 无股票可选，则直接进行下一次循环
            if curr_factor_data.dropna().empty:
                continue
            # 对因子值进行调整，使得其在qcut后，分组标签越小的总是在最有利的方向上
            if direction is '+':
                curr_factor_data = -curr_factor_data
            elif direction is '-':
                pass
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
                
            # 进行qcut
            labeled_factor = pd.qcut(curr_factor_data, no_of_groups, labels = False)
            # 选取指定组的股票，注意标签是0开始，传入参数是1开始，因此需要减1
            selected_stocks = curr_factor_data.ix[labeled_factor == group-1].index
            # 被选取股票的持仓调为1
            self.position.holding_matrix.ix[time, selected_stocks] = 1
            
        # 循环结束
        # 去除不可交易的股票
        self.filter_untradable()
        # 设置为等权重
        self.position.to_percentage()
        # 如果需要市值加权，则市值加权
        if holding == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['MarketValue',
                                           self.position.holding_matrix.index, :])
                
        
        
        
    
            
            
            
                
                
            
            
            
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            