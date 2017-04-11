#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:10:54 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from backtest_data import backtest_data
from position import position
from performance import performance
from performance_attribution import performance_attribution

# 回测类，对给定的持仓进行回测

class backtest(object):
    """ The class for backtest.
    
    foo
    """
    
    def __init__(self, bkt_position, *, initial_money = 100000000, trade_ratio = 0.95, 
                 buy_cost = 1.3/1000, sell_cost = 1.3/1000, bkt_start = 'default', bkt_end = 'default', 
                 risk_free_rate = 0.05, bkt_stock_data = 'default', bkt_benchmark_data = 'default'):
        """ Initialize backtest object.
        
        foo
        """
        # 初始化传入的持仓类，是要回测的策略构造出的持仓矩阵对象，是回测的目标持仓，注意此日期为调仓日
        self.bkt_position = bkt_position
        
        # 初始化回测用到的股价数据类
        self.bkt_data = backtest_data()
        # 初始化股价数据，包括收盘开盘价等
        if bkt_stock_data == 'default':
            self.bkt_data.stock_price = data.read_data(['ClosePrice_adj','OpenPrice_adj'], 
                                                  ['ClosePrice_adj','OpenPrice_adj'])
        else:
            self.bkt_data.stock_price = data.read_data(bkt_stock_data)
        # 初始化基准价格数据，默认设为中证500，只需要收盘数据
        if bkt_benchmark_data == 'default':
            self.bkt_data.benchmark_price = data.read_data(['ClosePrice_zz500','OpenPrice_zz500'], 
                                                      ['ClosePrice','OpenPrice'])
        else:
            self.bkt_data.benchmark_price = data.read_data([bkt_benchmark_data], [bkt_benchmark_data])
        # 读取股票上市退市停牌数据，并生成标记股票是否可交易的矩阵
        self.bkt_data.generate_if_tradable()
            
        # 根据传入的持仓类，校准回测股价和基准股价的数据，将股票代码对齐
        self.bkt_data.stock_price = data.align_index(self.bkt_position.holding_matrix, self.bkt_data.stock_price, 
                                                     axis = 'minor')
        self.bkt_data.if_tradable = data.align_index(self.bkt_position.holding_matrix, self.bkt_data.if_tradable, 
                                                     axis = 'minor')
        
        # 检测股票代码是否都包含在回测数据中，当有一只股票的某一个回测数据全是nan，且对这只股票有持仓时，
        # 则认为有股票代码没有全部包含在回测数据中
        stock_in_condition = np.logical_and(self.bkt_data.stock_price.isnull().all(1).any(1),
                                            self.bkt_position.holding_matrix.sum()>0)
        assert not stock_in_condition.any(), \
               'Some stocks in the input holding matrix are NOT included in the backtest database, '\
               'please check it carefully!\n'
        # 检测回测数据是否覆盖了回测时间段
        # 检测起始时间
        if bkt_start == 'default':
            assert self.bkt_data.stock_price.major_axis[0]<=self.bkt_position.holding_matrix.index[0], \
                   'The default start time of backtest is earlier than the start time in backtest database, '\
                   'please try to set a later start time which must be a trading day\n'
        else:
            assert self.bkt_data.stock_price.major_axis[0]<=bkt_start, \
                   'The input start time of backtest is earlier than the start time in backteset database, '\
                   'please try to set a later start time which must be a trading day, or try to set it as default\n'
        # 检测结束时间
        if bkt_end == 'default':
            # 如果回测数据中的最后一天直接在最后一个调仓日前，则直接报错
            assert self.bkt_data.stock_price.major_axis[-1]>self.bkt_position.holding_matrix.index[-1], \
                   'The default end time of backtest is later than the end time in backtest database, '\
                   'please try to set an earlier end time which must be a trading day\n'
            # 回测数据中的最后一天在最后一个调仓日后，现在判断是否之后有60个交易日可取
            last_holding_loc = self.bkt_data.stock_price.major_axis.get_loc(self.bkt_position.holding_matrix.index[-1])
            total_size = self.bkt_data.stock_price.major_axis.size
            assert total_size>=last_holding_loc+1+60, \
                   'The default end time of backtest is later than the end time in backtest database, '\
                   'please try to set an earlier end time which must be a trading day\n'
        else:
            assert self.bkt_data.stock_price.major_axis[-1]>bkt_end, \
                   'The input end time of backtest is later than the end time in backtest database, '\
                   'please try to set an earlier end time which must be a trading day, or try to set it as default\n'
        
        # 设置回测的起止时间，这里要注意默认的时间可能超过回测数据的范围
        # 起始时间：默认为第一个调仓日，如有输入数据，则为输入数据和默认时间的较晚日期
        default_start = self.bkt_data.stock_price.major_axis[self.bkt_data.stock_price.major_axis.get_loc(self.bkt_position.holding_matrix.index[0])]
        if bkt_start == 'default':
            self.bkt_start = default_start
        else:
            self.bkt_start = max(default_start, bkt_start)
        # 停止时间：默认为最后一个调仓日后的21个交易日，如有输入数据，则以输入数据为准
        if bkt_end == 'default':
            default_end = self.bkt_data.stock_price.major_axis[self.bkt_data.stock_price.major_axis.get_loc(self.bkt_position.holding_matrix.index[-1])+21]
            self.bkt_end = default_end
        else:
            self.bkt_end = bkt_end
            
        # 对回测的其他数据进行初始化
        self.initial_money = initial_money
        self.trade_ratio = trade_ratio
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.risk_free_rate = risk_free_rate
        
        # 以回测期（而不是回测数据期或调仓期）为时间索引的持仓量矩阵，注意vol的持仓单位为手，pct的持仓单位为百分比
        start_loc = self.bkt_data.stock_price.major_axis.get_loc(self.bkt_start)
        end_loc = self.bkt_data.stock_price.major_axis.get_loc(self.bkt_end)
        backtest_period_holding_matrix = self.bkt_data.stock_price.ix[0,start_loc:end_loc+1,:]
        self.tar_pct_position = position(backtest_period_holding_matrix)
        # 初始化持仓目标矩阵
        self.tar_pct_position.holding_matrix = self.bkt_position.holding_matrix.reindex(
                                               index = self.tar_pct_position.holding_matrix.index, 
                                               method = 'ffill')
        # 初始化实际持仓矩阵
        self.real_vol_position = position(backtest_period_holding_matrix)
        # 初始化实际持仓的百分比
        self.real_pct_position = position(backtest_period_holding_matrix)
        # 初始化目标持仓矩阵，单位为手，这个持仓量矩阵主要作为参考
        self.tar_vol_position = position(backtest_period_holding_matrix)
        
        # 将回测数据期也调整为回测期
        self.bkt_data.stock_price = data.align_index(self.tar_pct_position.holding_matrix, self.bkt_data.stock_price, 
                                                     axis = 'major')
        self.bkt_data.benchmark_price = data.align_index(self.tar_pct_position.holding_matrix, self.bkt_data.benchmark_price, 
                                                         axis = 'major')
        self.bkt_data.if_tradable = data.align_index(self.tar_pct_position.holding_matrix, self.bkt_data.if_tradable, 
                                                     axis = 'major')
        
        # 初始化回测要用到的现金数据：
        self.cash = pd.Series(np.zeros(self.real_vol_position.holding_matrix.shape[0]), 
                                 index = self.real_vol_position.holding_matrix.index)
        self.cash.ix[0] = self.initial_money*self.trade_ratio
        # 初始化回测得到的账户价值数据：
        self.account_value = []
        # 初始化计算业绩指标及作图用到的benchmark价值数据
        self.benchmark_value = self.bkt_data.benchmark_price.ix['ClosePrice', :, 0]
        
        # 暂时用一个警告的string初始化performance对象，防止提前调用此对象出错
        self.bkt_performance = 'The performance object of this backtest object has NOT been initialized, '\
                               'please try to call this attribute after call backtest.get_performance()\n'

        # 显示回测是否重置的标签，多次用到同一个回测对象时，用其控制回测数据的重置
        self.is_bkt_reset = True

        # 初始化结束时，目标和实际持仓矩阵、回测数据都是一样的时间股票索引（即策略持仓股票为股票索引，回测期间为时间索引），
        # 传入的bkt_position股票索引是一样的，但是时间索引为调仓日的时间
        print('The backtest system has been successfully initialized!\n')
        
    def execute_backtest(self):
        """ Execute the backtest.
        
        foo
        """
        self.is_bkt_reset = False
        cursor = -1
        # 开始执行循环，对tar_pct_position.holding_matrix进行循环
        for curr_time, curr_tar_pct_holding in self.tar_pct_position.holding_matrix.iterrows():
            
            cursor += 1
            
            # 如为回测第一天
            if cursor == 0:
                self.deal_with_first_day(curr_time, curr_tar_pct_holding)
            
            # 非回测第一天
            # 如果为非调仓日
            elif curr_time not in self.bkt_position.holding_matrix.index:
                
                self.real_vol_position.holding_matrix.ix[cursor, :] = self.real_vol_position.holding_matrix.ix[cursor-1, :]
                self.cash.ix[cursor] = self.cash.ix[cursor-1]
                
                # 处理当日退市的股票
                self.deal_with_held_delisted(curr_time, cursor)
        
            # 如果为调仓日
            else:
                
                self.cash.ix[cursor] = self.cash.ix[cursor-1]
                
                # 首先必须有对当天退市股票的处理
                self.deal_with_held_delisted(curr_time, cursor)
                
                # 计算预计持仓量矩阵，以确定当期的交易计划
                proj_vol_holding = self.get_proj_vol_holding(curr_tar_pct_holding, cursor)
                
                # 根据预计持仓矩阵，进行实际交易
                self.execute_real_trading(curr_time, cursor, proj_vol_holding)
                
        # 循环结束，开始计算持仓的序列
        self.real_pct_position.holding_matrix = self.real_vol_position.holding_matrix.div(self.real_vol_position.holding_matrix.sum(1), 
                                                                                         axis = 0)
        
        # 计算账面的价值，注意，这里的账面价值没有加上资金中不能用于投资的部分（即1-trade_ratio那部分）
        self.account_value = (self.real_vol_position.holding_matrix * 100 * \
                              self.bkt_data.stock_price.ix['ClosePrice_adj', :, :]).sum(1) + \
                              self.cash
                              
        # 我们的账面价值序列，如果第一天就调仓（默认就是这种情况），最开始会不是初始资金，因此在第一行加入初始资金行
        # 初始资金这一行的时间设定为回测开始时间的前一秒
        base_time = self.bkt_start - pd.tseries.offsets.Second(1)
        base_value = pd.Series(self.initial_money * self.trade_ratio, index = [base_time])
        # 拼接在一起
        self.account_value = pd.concat([base_value, self.account_value])
        # 拼接benchmark价值序列，初始值则用回测开始当天的开盘价
        benchmark_base_value = pd.Series(self.bkt_data.benchmark_price.ix['OpenPrice', 0, 0], 
                                         index = [base_time])
        self.benchmark_value = pd.concat([benchmark_base_value, self.benchmark_value])
    
    # 单独处理回测的第一期，因为这一期没有cursor-1项
    def deal_with_first_day(self, curr_time, curr_tar_pct_holding):
        
        # 如果为非调仓日
        if curr_time not in self.bkt_position.holding_matrix.index:
            # 实际持仓本来就被初始化为0，也没有需要处理的退市股票，因此这里暂时什么也不做
            pass
        
        # 如果为调仓日，默认情况下的回测第一天即为调仓第一天，是调仓日
        else: 
            # 并没有要处理的退市股票，也没有要卖的股票，预计可以使用的资金就是全部资金，预计买入的量就是要买入的量，因此直接用所有资金买入预计要买入的量
            
            # 可以交易的股票，即那些已上市，未退市，未停牌的股票
            tradable = self.bkt_data.if_tradable.ix['if_tradable', 0, :]
            
            if not curr_tar_pct_holding.ix[tradable].empty:
                # 对可买入的股票进行权重的重新归一计算，直接就用这个百分比买入股票
                tradable_pct = curr_tar_pct_holding.ix[tradable] / (curr_tar_pct_holding.ix[tradable]).sum()
                # 计算买入的量
                real_buy_vol = np.floor(self.cash.ix[0] * tradable_pct * (1-self.buy_cost) / \
                                        (self.bkt_data.stock_price.ix['OpenPrice_adj', 0, :] * 100))
                # 买入后的资金
                self.cash.ix[0] -= (real_buy_vol / (1-self.buy_cost) * 100 * \
                                    self.bkt_data.stock_price.ix['OpenPrice_adj', 0, :]).sum()
                # 买入后的持仓
                self.real_vol_position.add_holding(curr_time, real_buy_vol)
     
    # 处理持有的当日退市的股票
    def deal_with_held_delisted(self, curr_time, cursor):
        # 如果实际持仓中有当日退市的股票，则以上一个交易日的收盘价卖掉这些股票，这里计算了交易费
        vol_held_delisted = self.real_vol_position.holding_matrix.ix[cursor] * \
                              self.bkt_data.if_tradable.ix['is_delisted',cursor,:] * \
                              np.logical_not(self.bkt_data.if_tradable.ix['is_delisted',cursor-1,:])
        # 卖掉股票
        self.real_vol_position.subtract_holding(curr_time, vol_held_delisted)
        # 计算得到的现金
        self.cash.ix[cursor] += (self.bkt_data.stock_price.ix['ClosePrice_adj', cursor-1, :] * \
                                 vol_held_delisted * 100 * (1-self.sell_cost)).sum()
        
    # 计算预计持仓量矩阵，即预计的要持有的股票数量（单位：手）
    def get_proj_vol_holding(self, curr_tar_pct_holding, cursor):
        # 预估要买入的量，先预估卖出可卖出的股票后的资金量
        # 可以交易的股票，即那些已上市，未退市，未停牌的股票
        tradable = self.bkt_data.if_tradable.ix['if_tradable',cursor,:]
                           
        # 以当期的开盘价，卖出上一期持有的可以交易的股票，加上之前的可用现金，得到当期可用的资金
        # 预估交易和此后的实际交易中，股票买卖价格均为开盘价，即假设开盘时一瞬间，就计算出了预计交易量和进行了实际交易
        # 另一个关于此的假设是，在实际交易中，会考虑到涨跌停问题，但是在预估交易时不用到涨跌停信息，即使可能只用开盘价也能得到这些信息，这里日后可进行调整
        # 这里预估的时候，卖价没有计算交易费用，这样会导致对当期可用资金的高估，从而高估预计买入的量，因此这里还需要日后调整
        curr_cash_available = (self.real_vol_position.holding_matrix.ix[cursor-1, tradable] * \
                               self.bkt_data.stock_price.ix['OpenPrice_adj', cursor, tradable] * \
                               100 ).sum() + self.cash.ix[cursor-1]
                                                       
        # 对目标持仓股票中，可以交易的股票进行权重的重新归一计算
        projected_pct = curr_tar_pct_holding.ix[tradable] / (curr_tar_pct_holding.ix[tradable]).sum()
                
        # 计算预计买入的量，注意这里依然不计算交易费用
        projected_vol = np.floor(pd.to_numeric(curr_cash_available * projected_pct /
                                 (self.bkt_data.stock_price.ix['OpenPrice_adj', cursor, tradable] *100)))
        
        # 预计的当期新持仓量向量，注意这里与上面的不同在于这里包含所有股票的代码
        #proj_vol_holding = pd.Series(self.real_vol_position.holding_matrix.ix[cursor-1, :]
        #proj_vol_holding.ix[tradable] = projected_vol
        proj_vol_holding = projected_vol.reindex(self.real_vol_position.holding_matrix.columns, fill_value=0)
        
        return proj_vol_holding
        
    # 根据预计持仓量，进行真实的交易
    def execute_real_trading(self, curr_time, cursor, proj_vol_holding):
        # 预计的交易量，即交易计划，大于0为买入，小于0为卖出
        trade_plan = proj_vol_holding - self.real_vol_position.holding_matrix.ix[cursor-1,:]
                
        # 开始真正的交易，先卖后买
        # 首先，将上一期的持仓移动到这一期
        self.real_vol_position.holding_matrix.ix[cursor, :] = self.real_vol_position.holding_matrix.ix[cursor-1, :]
                
        # 处理卖出
        sell_plan = -(trade_plan.ix[trade_plan<0])
        # 有卖出
        if not sell_plan.empty:
            # 卖出后的资金
            self.cash.ix[cursor] += (sell_plan * self.bkt_data.stock_price.ix['OpenPrice_adj', cursor, :] * \
                                     100 * (1-self.sell_cost)).sum()
            # 卖出后的持仓
            self.real_vol_position.subtract_holding(curr_time, sell_plan)
                
        # 处理买入
        buy_plan = trade_plan.ix[trade_plan>0]
        # 有买入
        if not buy_plan.empty:
            # 计算买入量的百分比，这是因为，有实际操作以及刚刚提到的交易费用的原因，计划的买入量和实际的买入量会不同，只能按比例买
            buy_plan_pct = buy_plan / buy_plan.sum()
            # 实际买入的量，用实际的现金，以buy_plan的比例买入股票
            real_buy_vol = np.floor(self.cash.ix[cursor] * buy_plan_pct * (1-self.buy_cost) / \
                                    (self.bkt_data.stock_price.ix['OpenPrice_adj', cursor, :] * 100))
            # 买入后的资金
            self.cash.ix[cursor] -= (real_buy_vol / (1 - self.buy_cost) * 100 * \
                                     self.bkt_data.stock_price.ix['OpenPrice_adj', cursor, :]).sum()
            # 买入后的持仓
            self.real_vol_position.add_holding(curr_time, real_buy_vol)
            
    # 仅仅初始化performance类，只得到净值和收益数据，而不输出指标和画图
    def initialize_performance(self):
        self.bkt_performance = performance(self.account_value, benchmark = self.benchmark_value, 
                                  risk_free_rate = self.risk_free_rate) 
            
    # 计算回测得到的收益率数据，得到业绩指标以及绘图
    def get_performance(self):
        # 初始化performance对象
        self.initialize_performance()
        
        # 计算和输出业绩指标
        self.bkt_performance.get_performance()
        # 画图
        self.bkt_performance.plot_performance()

    # 利用回测得到的数据，或直接算出的数据进行业绩归因
    def get_performance_attribution(self, *, benchmark_position='default', outside_bb='Empty', discard_factor=[],
                                    show_warning=True, is_real_world=False):
        if is_real_world:
            self.bkt_pa = performance_attribution(self.real_pct_position, benchmark_position=benchmark_position,
                                                  portfolio_returns=self.bkt_performance.log_return)
        else:
            self.bkt_pa = performance_attribution(self.tar_pct_position, benchmark_position=benchmark_position,
                                                  )
        self.bkt_pa.execute_performance_attribution(outside_bb=outside_bb, discard_factor=discard_factor,
                                                    show_warning=show_warning)
        self.bkt_pa.plot_performance_attribution()

    # 重置回测每次执行回测要改变的数据，若想不创建新回测对象而改变回测参数，则需重置这些数据后才能再次执行回测
    def reset_bkt_data(self):
        if not self.is_bkt_reset:
            # 重置现金序列，账户序列以及benchmark序列
            self.cash = pd.Series(np.zeros(self.real_vol_position.holding_matrix.shape[0]),
                                  index=self.real_vol_position.holding_matrix.index)
            self.cash.ix[0] = self.initial_money * self.trade_ratio
            self.account_value = []
            self.benchmark_value = self.bkt_data.benchmark_price.ix['ClosePrice', :, 0]

            self.is_bkt_reset = True

    # 重置传入的持仓矩阵参数的函数，当要测试同一个策略的不同参数对其的影响时，会用到，这样可以不必重新创建一个回测对象
    # 注意这里只改变了传入的持仓矩阵，包括回测时间，股票id，benchmark等其余参数一律不变
    def reset_bkt_position(self, new_bkt_position):
        self.bkt_position = new_bkt_position
        # 重新将目标持仓，实际持仓等矩阵初始化
        self.tar_pct_position.holding_matrix = self.bkt_position.holding_matrix.reindex(
                                               index = self.tar_pct_position.holding_matrix.index, 
                                               method = 'ffill')
        self.real_vol_position = position(self.tar_pct_position.holding_matrix)
        self.real_pct_position = position(self.tar_pct_position.holding_matrix)
        self.tar_vol_position = position(self.tar_pct_position.holding_matrix)

        # 重置回测数据
        self.reset_bkt_data()

    # 重置benchmark，需要观察一个策略相对不同benchmark的变化时用到，包括改变股票池后，benchmark应当换成对应的股票池
    def reset_bkt_benchmark(self, new_bkt_benchmark_data):
        self.bkt_data.benchmark_price = data.read_data(new_bkt_benchmark_data, ['ClosePrice','OpenPrice'])

        # 将benchmark price数据期调整为回测期
        self.bkt_data.benchmark_price = data.align_index(self.tar_pct_position.holding_matrix,
                                                         self.bkt_data.benchmark_price, axis='major')

        # 重置回测数据
        self.reset_bkt_data()



                
                
                
                           
            
            
                        
                
                
                
                
                
                
                
        
        

        
        
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    