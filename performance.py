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
    
    def __init__(self, account_value, *, benchmark = pd.Series(), holding_days='default', info_series='default',
                 tradedays_one_year = 252, risk_free_rate = 0.0):
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
        # 策略的调仓日，在有benchmark，计算策略的超额净值的时候会用到
        self.holding_days = holding_days
        # 其他信息，包括换手率，持股数等
        self.info_series = info_series
    
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

            # 超额净值，注意超额净值并不是账户净值减去基准净值，因为超额净值要考虑到策略在调仓日对基准份额的调整
            # 超额净值的算法为，每个调仓周期之内的超额净值序列为exp（策略累计收益序列）- exp（基准累计收益序列）
            # 不同调仓周期之间的净值为：这个调仓周期内的超额净值序列加上上一个调仓周期的最后一天的净值
            return_data = pd.DataFrame({'log_return': self.log_return, 'log_return_bench': self.log_return_bench})
            return_data['mark'] = self.holding_days.asof(return_data.index).replace(pd.tslib.NaT, account_value.index[0])
            grouped = return_data.groupby('mark')
            # 每个调仓周期内用周期内净值相减的方法
            intra_holding = grouped.apply(lambda x:
                np.exp(x['log_return'].cumsum()) - np.exp(x['log_return_bench'].cumsum())).reset_index(0, drop=True)
            # 算每个调仓周期的最后一天的周期内净值
            holding_node_value = grouped.apply(
                lambda x: np.exp(x['log_return'].sum()) - np.exp(x['log_return_bench'].sum()))
            # 此后的每个周期内的净值，都需要加上此前所有周期的最后一天的净值，注意首先需要shift一个调仓周期
            # 因为每个调仓周期最后的净值，要到下一个周期内才加上
            holding_node_value_cum = holding_node_value.shift(1).cumsum().fillna(0). \
                reindex(intra_holding.index, method='ffill')
            self.excess_net_account_value = holding_node_value_cum + intra_holding
            self.excess_net_account_value += 1
            # 计算用超额净值得到的超额收益序列，用这个序列来计算超额收益的统计量，更符合实际
            self.excess_nv_return = np.log(self.excess_net_account_value/self.excess_net_account_value.shift(1))
            self.cum_excess_nv_return = self.excess_nv_return.cumsum()

            
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

    # 计算年化calmar比率
    @staticmethod
    def annual_calmar_ratio(annual_return, max_drawdown):
        return annual_return / max_drawdown

    # 计算年化sortino比率
    @staticmethod
    def annual_sortino_ratio(return_series, annual_return, *, return_target=0.0,
                             tradedays_one_year=252, risk_free_rate=0.0):
        under_performance_return = return_series - return_target
        under_performance_return = under_performance_return.where(under_performance_return<0, 0.0)
        sortino = (annual_return - risk_free_rate) / (under_performance_return.std() * np.sqrt(tradedays_one_year))
        return sortino
        
    # 接下来计算与benchmark相关的指标
    
    # 年化超额收益
    def annual_excess_return(self):
        return self.cum_excess_nv_return.ix[-1] * self.tradedays_one_year / \
               (self.cum_excess_nv_return.size - 1)
               
    # 年化超额收益波动率
    def annual_excess_std(self):
        return self.excess_nv_return.std() * np.sqrt(self.tradedays_one_year)
        
    # 年化信息比
    def info_ratio(self, annual_excess_return, annual_excess_std):
        return annual_excess_return / annual_excess_std
        
    # 胜率
    def win_ratio(self):
        return self.excess_nv_return.ix[self.excess_nv_return>0].size / \
               self.excess_nv_return.size
        
    # 计算并输出各个指标
    def get_performance(self, *, foldername=''):
        
        annual_r = performance.annual_return(self.cum_log_return, self.tradedays_one_year)
        annual_std = performance.annual_std(self.log_return, self.tradedays_one_year)
        annual_sharpe = performance.annual_sharpe(annual_r, annual_std, self.risk_free_rate)
        max_dd, peak_loc, low_loc = performance.max_drawdown(self.account_value)
        annual_calmar = performance.annual_calmar_ratio(annual_r, max_dd)
        annual_sortino = performance.annual_sortino_ratio(self.log_return, annual_r, return_target=0.0,
                            tradedays_one_year=self.tradedays_one_year, risk_free_rate=self.risk_free_rate)
        annual_ex_r = self.annual_excess_return()
        annual_ex_std = self.annual_excess_std()
        annual_info_ratio = self.info_ratio(annual_ex_r, annual_ex_std)
        max_dd_ex, peak_loc_ex, low_loc_ex = performance.max_drawdown(self.excess_net_account_value)
        annual_ex_calmar = performance.annual_calmar_ratio(annual_ex_r, max_dd_ex)
        win_ratio = self.win_ratio()

        # 输出指标
        target_str = 'Stats START ------------------------------------------------------------------------\n' \
                     'The stats of the strategy (and its performance against benchmark) is as follows:\n' \
                     'Annual log return: {0:.2f}%\n' \
                     'Annual standard deviation of log return: {1:.2f}%\n' \
                     'Annual Sharpe ratio: {2:.2f}\n' \
                     'Max drawdown: {3:.2f}%\n' \
                     'Max drawdown happened between {4} and {5}\n' \
                     'Annual Calmar ratio: {6:.2f}\n' \
                     'Annual Sortino ratio: {7:.2f}\n' \
                     'Annual excess log return: {8:.2f}%\n' \
                     'Annual standard deviation of excess log return: {9:.2f}%\n' \
                     'Annual information ratio: {10:.2f}\n' \
                     'Max drawdown of excess account value: {11:.2f}%\n' \
                     'Max drawdown happened between {12} and {13}\n' \
                     'Annual excess Calmar ratio: {14:.2f}\n' \
                     'Winning ratio: {15:.2f}%\n' \
                     'Average turnover ratio: {16:.2f}%\n'\
                     'Average number of stocks holding: {17:.2f}\n'\
                     'Stats END --------------------------------------------------------------------------\n'.format(
            annual_r*100, annual_std*100, annual_sharpe, max_dd*100, self.cum_log_return.index[peak_loc],
            self.cum_log_return.index[low_loc], annual_calmar, annual_sortino, annual_ex_r*100,
            annual_ex_std*100, annual_info_ratio, max_dd_ex*100, self.cum_log_return.index[peak_loc_ex],
            self.cum_log_return.index[low_loc_ex], annual_ex_calmar, win_ratio*100,
            self.info_series.ix[:, 'turnover_ratio'].replace(0, np.nan).mean()*100,
            self.info_series.ix[:, 'holding_num'].mean()
        )

        print(target_str)

        # 将输出写到txt中
        with open(str(os.path.abspath('.'))+'/'+foldername+'/performance.txt',
                  'w', encoding='GB18030') as text_file:
            text_file.write(target_str)


    # 画图
    def plot_performance(self, *, foldername='', pdfs='default'):
        
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
        plt.savefig(str(os.path.abspath('.'))+'/'+foldername+'/CumLog.png', dpi=1200)
        if type(pdfs) != str:
            plt.savefig(pdfs, format='pdf')
        
        # 第二张图为策略超额收益曲线，只有在有benchmark的时候才画
        if not self.benchmark.empty:
            f2 = plt.figure()
            ax2 = f2.add_subplot(1,1,1)
            plt.plot(self.cum_excess_return*100, 'b-')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Log Return (%)')
            ax2.set_title('The Cumulative Excess Log Return of The Strategy')

            plt.savefig(str(os.path.abspath('.')) + '/' +foldername+'/ActiveCumLog.png', dpi=1200)
            if type(pdfs) != str:
                plt.savefig(pdfs, format='pdf')
            
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
        plt.savefig(str(os.path.abspath('.')) + '/' +foldername+'/NetValue.png', dpi=1200)
        if type(pdfs) != str:
            plt.savefig(pdfs, format='pdf')
        
        # 第四张图为策略超额收益净值，只有在有benchmark的时候才画
        if not self.benchmark.empty:
            f4 = plt.figure()
            ax4 = f4.add_subplot(1,1,1)
            plt.plot(self.excess_net_account_value, 'b-')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Excess Net Value')
            ax4.set_title('The Excess Net Value of The Strategy')
            plt.savefig(str(os.path.abspath('.')) + '/'+foldername+'/ActiveNetValue.png', dpi=1200)
            if type(pdfs) != str:
                plt.savefig(pdfs, format='pdf')

        # 第五张图画策略的持股数曲线
        f5 = plt.figure()
        ax5 = f5.add_subplot(1,1,1)
        plt.plot(self.info_series.ix[:, 'holding_num'], 'b-')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Number of Stocks')
        ax5.set_title('The Number of Stocks holding of The Strategy')
        plt.savefig(str(os.path.abspath('.')) + '/'+foldername+'/NumStocksHolding.png', dpi=1200)
        plt.savefig(pdfs, format='pdf')

            
            
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
