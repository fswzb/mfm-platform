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
import copy

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from backtest import backtest
from barra_base import barra_base


# 单因子表现测试

class single_factor_strategy(strategy):
    """ Single factor test strategy.
    
    foo
    """
    def __init__(self):
        strategy.__init__(self)
        self.strategy_data.generate_if_tradable(shift = True)
        # 读取市值数据以进行市值加权
        self.strategy_data.stock_price = data.read_data(['FreeMarketValue'],['FreeMarketValue'],shift = True)
        
    # 读取因子数据的函数
    def read_factor_data(self, file_name, factor_name, *, shift = True):
        self.strategy_data.factor = data.read_data(file_name, factor_name, shift = shift)

    # 生成调仓日的函数
    # holding_freq为持仓频率，默认为月，这个参数将作为resample的参数
    # start_date和end_date为调仓日的范围区间，默认为取数据的所有区间断
    def generate_holding_days(self, *, holding_freq='m', start_date='default', end_date='default'):
        # 读取free market value以其日期作为holding days的选取区间
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price.\
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 根据传入参数截取需要的调仓日区间
        if start_date != 'default':
            holding_days = holding_days.ix[start_date:]
        if end_date != 'default':
            holding_days = holding_days.ix[:end_date]
        self.holding_days = holding_days
        
    # 选取股票，选股比例默认为最前的80%到100%，方向默认为因子越大越好，weight=1为市值加权，0为等权
    def select_stocks(self, *, select_ratio = [0.8, 1], direction = '+', weight = 0):
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
        if self.strategy_data.stock_pool == 'all':
            # 去除不可交易的股票
            self.filter_untradable()
        else:
            # 有股票池的情况去除不可投资的股票
            self.filter_uninv()
        # 设置为等权重
        self.position.to_percentage()
        # 如果需要市值加权，则市值加权
        if weight == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
    
    # 单因子的因子收益率计算和检验，用来判断因子有效性，
    # holding_freq为回归收益的频率，默认为月，可调整为与调仓周期一样，也可不同
    # weights为用来回归的权重，默认为等权回归
    def get_factor_return(self, *, holding_freq='m', weights='default', direction='+'):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算因子收益的频率
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price.\
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 计算股票对数收益以及因子暴露
        holding_day_price = self.strategy_data.stock_price.ix['ClosePrice_adj',holding_days,:]
        holding_day_return = np.log(holding_day_price.div(holding_day_price.shift(1)))
        holding_day_factor = self.strategy_data.factor.ix[0, holding_days, :]
        holding_day_factor_expo = strategy_data.get_cap_wgt_exposure(holding_day_factor,
                                    self.strategy_data.stock_price.ix['FreeMarketValue', holding_days, :])
        # 注意因子暴露要用前一期的数据
        holding_day_factor_expo = holding_day_factor_expo.shift(1)

        # 初始化因子收益序列以及估计量的t统计量序列
        factor_return_series = np.empty(holding_days.size)*np.nan
        t_stats_series = np.empty(holding_days.size)*np.nan
        self.factor_return_series = pd.Series(factor_return_series, index=holding_days)
        self.t_stats_series = pd.Series(t_stats_series, index=holding_days)

        # 进行回归，对调仓日进行循环
        for cursor, time in holding_days.iteritems():

            y = holding_day_return.ix[time, :]
            x = holding_day_factor_expo.ix[time, :]
            if y.isnull().all() or x.isnull().all():
                continue
            x = sm.add_constant(x)
            if weights is 'default':
                results = sm.WLS(y, x, missing='drop').fit()
            else:
                results = sm.WLS(y, x, weights=weights.ix[time], missing='drop').fit()
            self.factor_return_series.ix[time] = results.params[1]
            self.t_stats_series.ix[time] = results.tvalues[1]

        # 如果方向为负，则将因子收益和t统计量加个负号
        if direction == '-':
            self.factor_return_series = -self.factor_return_series
            self.t_stats_series = -self.t_stats_series

        # 循环结束，输出结果
        print('The average return of this factor: %f%%\n' % (self.factor_return_series.mean()*100))
        print('Note that the return of factor is not annualized but corresponding to the holding days interval\n')
        print('The average t-statistics value: %f\n' % (self.t_stats_series.mean()))
        tstats_sig_ratio = self.t_stats_series[np.abs(self.t_stats_series)>=2].size / self.t_stats_series.size
        print('Ratio of t-stats whose absolute value >= 2: %f\n' % (tstats_sig_ratio))

        # 画图
        fx = plt.figure()
        ax = fx.add_subplot(1,1,1)
        plt.plot(self.factor_return_series*100, 'b-')
        zero_series = pd.Series(np.zeros(self.factor_return_series.shape), index = self.factor_return_series.index)
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return of The Factor (%)')
        ax.set_title('The Return Series of The Factor')

        fx = plt.figure()
        ax = fx.add_subplot(1, 1, 1)
        plt.plot(self.t_stats_series, 'b-')
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('T-Stats of The Factor Return')
        ax.set_title('The T-Stats Series of The Factor Return')
            
    # 计算因子的IC，股票收益率是以holding_freq为频率的的收益率，默认为月
    def get_factor_ic(self, *, holding_freq='m', direction = '+'):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算ic的频率
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price. \
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 初始化ic矩阵
        ic_series = np.empty(holding_days.size)*np.nan
        self.ic_series = pd.Series(ic_series, index = holding_days)
        # 计算股票对数收益，提取因子值，同样的，因子值要用前一期的因子值
        holding_day_price = self.strategy_data.stock_price.ix['ClosePrice_adj',holding_days,:]
        holding_day_return = np.log(holding_day_price.div(holding_day_price.shift(1)))
        holding_day_factor = self.strategy_data.factor.ix[0, holding_days, :]
        holding_day_factor = holding_day_factor.shift(1)
        # 对调仓日进行循环
        for cursor, time in holding_days.iteritems():
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
        plt.plot(self.ic_series, 'b-')
        # 画一条一直为0的图，以方便观察IC的走势是否显著不为0
        zero_series = pd.Series(np.zeros(self.ic_series.shape), index = self.ic_series.index)
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('IC of The Factor')
        ax.set_title('The IC Time Series of The Factor')
        
    # 根据分位数分组选股，用来画同一因子不同分位数分组之间的收益率对比，以此判断因子的有效性
    def select_qgroup(self, no_of_groups, group, *, direction = '+', weight = 0):
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
        if self.strategy_data.stock_pool == 'all':
            # 去除不可交易的股票
            self.filter_untradable()
        else:
            # 有股票池的情况去除不可投资的股票
            self.filter_uninv()
        # 设置为等权重
        self.position.to_percentage()
        # 如果需要市值加权，则市值加权
        if weight == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
    
    # 循环画分位数图与long short图的函数
    # 定义按因子分位数选股的函数，将不同分位数收益率画到一张图上，同时还会画long-short的图
    # value=1为画净值曲线图，value=2为画对数收益率图，weight=0为等权，=1为市值加权
    def plot_qgroup(self, bkt, no_of_groups, *, direction='+', value=1, weight=0):
        # 默认画净值曲线图
        if value == 1:
            # 先初始化图片
            f1 = plt.figure()
            ax1 = f1.add_subplot(1, 1, 1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Net Account Value')
            ax1.set_title('Net Account Value Comparison of Different Quantile Groups of The Factor')

            # 开始循环选股、画图
            for group in range(no_of_groups):
                # 选股
                self.reset_position()
                self.select_qgroup(no_of_groups, group + 1, direction=direction, weight=weight)

                # 回测
                bkt.reset_bkt_position(self.position)
                bkt.execute_backtest()
                bkt.initialize_performance()

                # 画图，注意，这里画净值曲线图，差异很小时，净值曲线图的差异更明显
                plt.plot(bkt.bkt_performance.net_account_value, label='Group %s' % str(group + 1))

                # 储存第一组和最后一组以画long-short收益图
                if group == 0:
                    long_series = bkt.bkt_performance.net_account_value
                elif group == no_of_groups - 1:
                    short_series = bkt.bkt_performance.net_account_value

            ax1.legend(loc='best')

            # 画long-short的图
            f2 = plt.figure()
            ax2 = f2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Net Account Value')
            ax2.set_title('Net Account Value of Long-Short Portfolio of The Factor')
            plt.plot(long_series - short_series)

        elif value == 2:
            # 先初始化图片
            f1 = plt.figure()
            ax1 = f1.add_subplot(1, 1, 1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Cumulative Log Return (%)')
            ax1.set_title('Cumulative Log Return Comparison of Different Quantile Groups of The Factor')

            # 开始循环选股、画图
            for group in range(no_of_groups):
                # 选股
                self.reset_position()
                self.select_qgroup(no_of_groups, group + 1, direction=direction, weight=weight)

                # 回测
                bkt.reset_bkt_position(self.position)
                bkt.execute_backtest()
                bkt.initialize_performance()

                # 画图，注意，这里画累积对数收益图，当差异很大时，累积对数收益图看起来更容易
                plt.plot(bkt.bkt_performance.cum_log_return * 100, label='Group %s' % str(group + 1))

                # 储存第一组和最后一组以画long-short收益图
                if group == 0:
                    long_series = bkt.bkt_performance.cum_log_return
                elif group == no_of_groups - 1:
                    short_series = bkt.bkt_performance.cum_log_return

            ax1.legend(loc='best')

            # 画long-short的图
            f2 = plt.figure()
            ax2 = f2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Log Return (%)')
            ax2.set_title('Cumulative Log Return of Long-Short Portfolio of The Factor')
            plt.plot((long_series - short_series) * 100)

    # 根据一个股票池进行一次完整的单因子测试的函数
    def single_factor_test(self, *, factor, direction='+', bkt_obj='Empty', bb_obj='Empty', pa_benchmark_position='default',
                           discard_factor=[], bkt_start='default', bkt_end='default'):
        # 如果传入的是str，则读取同名文件，如果是dataframe，则直接传入因子
        if type(factor) == str:
            self.read_factor_data([factor], [factor], shift=True)
        elif self.strategy_data.factor.empty:
            self.strategy_data.factor = pd.Panel({'factor_one':factor})
        else:
            self.strategy_data.factor[0] = factor

        # 生成调仓日
        if self.holding_days.empty:
            self.generate_holding_days()
        # 根据股票池生成标记
        self.strategy_data.handle_stock_pool(shift=True)
        # 除去不可交易或不可投资的数据
        if self.strategy_data.stock_pool == 'all':
            self.strategy_data.discard_untradable_data()
        else:
            self.strategy_data.discard_uninv_data()
        # 初始化持仓
        if self.position.holding_matrix.empty:
            self.initialize_position(self.strategy_data.factor.ix[0, self.holding_days, :])
        
        # 简单分位数选股
        self.select_stocks(weight=1, direction=direction)

        # 如果有外来的backtest对象，则使用这个backtest对象，如果没有，则需要自己建立，同时传入最新持仓
        if bkt_obj == 'Empty':
            bkt_obj = backtest(self.position, bkt_start=bkt_start, bkt_end=bkt_end)
        else:
            bkt_obj.reset_bkt_position(self.position)
        # 回测、画图、归因
        bkt_obj.execute_backtest()
        bkt_obj.get_performance()
        bkt_obj.get_performance_attribution(outside_bb=bb_obj, benchmark_position=pa_benchmark_position,
                                            discard_factor=discard_factor, show_warning=False)
        # 画单因子组合收益率
        self.get_factor_return(weights=np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue']),
                               holding_freq='d', direction=direction)
        # 画ic的走势图
        self.get_factor_ic(direction=direction, holding_freq='m')
        # 画分位数图和long short图
        self.plot_qgroup(bkt_obj, 5, direction=direction, value=1, weight=0)

    # 根据多个股票池进行一次完整的单因子测试
    def sf_test_multiple_pools(self, *, factor, direction='+', bb_obj='Empty', discard_factor=[],
                               stock_pools=['all', 'hs300', 'zz500', 'zz800'], bkt_start='default', bkt_end='default'):
        # 如果传入的是str，则读取同名文件，如果是dataframe，则直接传入因子
        # 注意：这里的因子数据并不储存到self.strategy_data.factor中，因为循环股票池会丢失数据
        # 这里实际上是希望每次循环都有一个新的single factor strategy对象，
        # 而这个函数的目的则是每次循环不用再重新建立这个对象
        if type(factor) == str:
            factor_data = data.read_data([factor], [factor], shift=True)
            factor = factor_data[factor]
        # 生成调仓日
        self.generate_holding_days()
        # 初始化持仓
        self.initialize_position(factor.ix[self.holding_days,:])
        # 先要初始化bkt对象
        bkt_obj = backtest(self.position, bkt_start=bkt_start, bkt_end=bkt_end)
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
            # 将回测的基准改为当前的股票池，若为all，则用默认的基准值
            if stock_pool != 'all':
                bkt_obj.reset_bkt_benchmark(['ClosePrice_'+stock_pool, 'OpenPrice_'+stock_pool])
            # 重置策略持仓
            self.reset_position()
            # 将策略的股票池设置为当前股票池
            self.strategy_data.stock_pool = stock_pool
            # 将bb的股票池改为当前股票池
            bb_obj.bb_data.stock_pool = stock_pool
            # 根据股票池生成标记，注意：股票池数据不需要shift，因为这里的barrabase数据是用来事后归因的，不涉及策略构成
            bb_obj.bb_data.handle_stock_pool(shift=False)

            # 进行当前股票池下的单因子测试
            # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
            self.single_factor_test(factor=factor, direction=direction, bkt_obj=bkt_obj, bb_obj=copy.deepcopy(bb_obj),
                                    discard_factor=discard_factor, bkt_start=bkt_start, bkt_end=bkt_end)




                
        
        
        
    
            
            
            
                
                
            
            
            
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            