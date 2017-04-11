#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:06:16 2016

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

# 表现类，即根据账户的时间序列，计算各种业绩指标，以及进行画图
class performance(object):
    """ The class for performance calculation and plot.
    
    foo
    """
    
    def __init__(self, account_value, *, benchmark = pd.Series(), 
                 tradedays_one_year = 252, risk_free_rate = 0.05):
        """ Initialize performance object.
        
        foo
        """
        self.account_value = account_value
        self.benchmark = benchmark
        self.tradedays_one_year = tradedays_one_year
        self.risk_free_rate = risk_free_rate
        # 储存一下第一行的时间点，这个点为虚拟的所用动作开始前的点，资金为原始资金
        base_timestamp = self.account_value.index[0]
        # 简单收益率，这时去除第一项，这种序列用来计算统计量
        self.simple_return = self.account_value.pct_change().ix[1:]
        # 对数收益率，同上
        self.log_return = (np.log(self.account_value/self.account_value.shift(1))).ix[1:]
        # 累积简单收益率，这种收益率用来画图，以及计算最大回撤等，注意这个收益率序列有起始项
        self.cum_simple_return = (self.simple_return+1).cumprod()-1
        # 累积对数收益
        self.cum_log_return = self.log_return.cumsum()
        # 拼接起始项
        base_series = pd.Series(0, index = [base_timestamp])
        self.cum_simple_return = pd.concat([base_series, self.cum_simple_return])
        self.cum_log_return = pd.concat([base_series, self.cum_log_return])
        
        # 策略账户净值序列
        self.net_account_value = self.account_value / self.account_value.ix[0]
    
        # 对benchmark进行同样的计算，暂时只管一个benchmark数据
        if not self.benchmark.empty:
            self.simple_return_bench = self.benchmark.pct_change().ix[1:]
            self.log_return_bench = (np.log(self.benchmark/self.benchmark.shift(1))).ix[1:]
            self.cum_simple_return_bench = (self.simple_return_bench+1).cumprod()-1
            self.cum_log_return_bench = self.log_return_bench.cumsum()
            self.cum_simple_return_bench = pd.concat([base_series, self.cum_simple_return_bench])
            self.cum_log_return_bench = pd.concat([base_series,self.cum_log_return_bench])
            
            # 计算超额累积收益序列以及超额日收益序列，均用对数收益序列计算
            self.cum_excess_return = self.cum_log_return - self.cum_log_return_bench
            self.excess_return = self.log_return - self.log_return_bench
            
            # benchmark的账户净值
            # 第一个不为nan的数
            bench_base_value = self.benchmark[self.benchmark.notnull()].ix[0]
            self.net_benchmark = self.benchmark / bench_base_value
            # 超额净值
            self.excess_net_value = self.net_account_value - self.net_benchmark
            
            
    # 定义各种计算指标的函数，这里都用对数收益来计算
    
    # 年化收益
    @staticmethod
    def annual_return(cum_return_series, tradedays_one_year):
        return cum_return_series.ix[-1] / (cum_return_series.size-1) * tradedays_one_year
        
    # 年化波动率
    @staticmethod
    def annual_std(return_series, tradedays_one_year):
        return return_series.std() * np.sqrt(tradedays_one_year)
        
    # 年化夏普比率
    @staticmethod
    def annual_sharpe(annual_return, annual_std, risk_free_rate):
        return (annual_return - risk_free_rate) / annual_std
        
    # 最大回撤率，返回值为最大回撤率，以及发生的时间点的位置
    @staticmethod
    def max_drawdown(account_value_series):
        past_peak = account_value_series.ix[0]
        max_dd = 0
        past_peak_loc = 0
        low_loc = 0
        for i, curr_account_value in enumerate(account_value_series):
            if curr_account_value >= past_peak:
                past_peak = curr_account_value
                temp_past_peak_loc = i
            elif (curr_account_value - past_peak) / past_peak < max_dd:
                max_dd = (curr_account_value - past_peak) / past_peak
                low_loc = i
                past_peak_loc = temp_past_peak_loc
        return max_dd, past_peak_loc, low_loc
        
    # 接下来计算与benchmark相关的指标
    
    # 年化超额收益
    def annual_excess_return(self):
        return self.cum_excess_return.ix[-1] * self.tradedays_one_year / \
               (self.cum_excess_return.size - 1) 
               
    # 年化超额收益波动率
    def annual_excess_std(self):
        return self.excess_return.std() * np.sqrt(self.tradedays_one_year)
        
    # 年化信息比
    def info_ratio(self, annual_excess_return, annual_excess_std):
        return annual_excess_return / annual_excess_std
        
    # 胜率
    def win_ratio(self):
        return self.excess_return.ix[self.excess_return>0].size / \
               self.excess_return.size
        
    # 计算并输出各个指标
    def get_performance(self, *, foldername='', filename=''):
        
        annual_r = performance.annual_return(self.cum_log_return, self.tradedays_one_year)
        annual_std = performance.annual_std(self.log_return, self.tradedays_one_year)
        annual_sharpe = performance.annual_sharpe(annual_r, annual_std, self.risk_free_rate)
        max_dd, peak_loc, low_loc = performance.max_drawdown(self.account_value)
        annual_ex_r = self.annual_excess_return()
        annual_ex_std = self.annual_excess_std()
        annual_info_ratio = self.info_ratio(annual_ex_r, annual_ex_std)
        win_ratio = self.win_ratio()

        # 输出指标
        target_str = 'Stats START ------------------------------------------------------------------------\n' \
                     'The stats of the strategy (and its performance agianst benchmark) is as follows:\n' \
                     'Annual log return: {0:.2f}%\n' \
                     'Annual standard deviation of log return: {1:.2f}%\n' \
                     'Anuual Sharpe ratio: {2:.2f}\n' \
                     'Max Drawdown: {3:.2f}%\n' \
                     'Max Drawdown happened between {4} and {5}\n' \
                     'Anuual excess log return: {6:.2f}%\n' \
                     'Anuual standard deviation of excess log return: {7:.2f}%\n' \
                     'Anuual information ratio: {8:.2f}\n' \
                     'Winning ratio: {9:.2f}%\n' \
                     'Stats END --------------------------------------------------------------------------\n'.format(
            annual_r*100, annual_std*100, annual_sharpe, max_dd*100, self.cum_log_return.index[peak_loc],
            self.cum_log_return.index[low_loc], annual_ex_r*100, annual_ex_std*100, annual_info_ratio, win_ratio*100
        )

        print(target_str)

        # 将输出写到txt中
        with open(str(os.path.abspath('.'))+'/'+foldername+'/'+filename+'performance.txt',
                  'w', encoding='GB18030') as text_file:
            text_file.write(target_str)


    # 画图
    def plot_performance(self, *, foldername='', filename=''):
        
        # 第一张图为策略自身累积收益曲线
        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.plot(self.cum_log_return*100, 'b-', label = 'Strategy')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Log Return (%)')
        ax1.set_title('The Cumulative Log Return of The Strategy (and The Benchmark)')
        
        # 如有benchmark，则加入benchmark的图
        if not self.benchmark.empty:
            plt.plot(self.cum_log_return_bench*100, 'r-', label = 'Benchmark')
            
        ax1.legend(loc = 'best')
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/'+filename+'CumLog.png')
        
        # 第二张图为策略超额收益曲线，只有在有benchmark的时候才画
        if not self.benchmark.empty:
            f2 = plt.figure()
            ax2 = f2.add_subplot(1,1,1)
            plt.plot(self.cum_excess_return*100, 'b-')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Log Return (%)')
            ax2.set_title('The Cumulative Excess Log Return of The Strategy')

            plt.savefig(str(os.path.abspath('.')) + '/' +foldername+'/'+ filename + 'ActiveCumLog.png')
            
        # 第三张图为策略账户净值曲线
        f3 = plt.figure()
        ax3 = f3.add_subplot(1,1,1)
        plt.plot(self.net_account_value, 'b-', label = 'Strategy')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Net Account Value')
        ax3.set_title('The Net Account Value of The Strategy (and The Benchmark)')
        
        # 如有benchmark，则加入benchmark的图
        if not self.benchmark.empty:
            plt.plot(self.net_benchmark, 'r-', label = 'Benchmark')
            
        ax3.legend(loc = 'best')
        plt.savefig(str(os.path.abspath('.')) + '/' +foldername+'/'+ filename + 'NetValue.png')
        
        # 第四张图为策略超额收益净值，只有在有benchmark的时候才画
        if not self.benchmark.empty:
            f4 = plt.figure()
            ax4 = f4.add_subplot(1,1,1)
            plt.plot(self.excess_net_value, 'b-')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Excess Net Value')
            ax4.set_title('The Excess Net Value of The Strategy')
            plt.savefig(str(os.path.abspath('.')) + '/'+foldername+'/' + filename + 'ActiveNetValue.png')
            
            
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
