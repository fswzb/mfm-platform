#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:51:44 2017

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

from single_factor_strategy import single_factor_strategy
from database import database
from data import data
from strategy_data import strategy_data

# 分析师预测覆盖因子的单因子策略

class analyst_coverage(single_factor_strategy):
    """Analyst coverage single factor strategy class.
    
    foo
    """
    def __init__(self):
        single_factor_strategy.__init__(self)
        # 该策略用于取数据的database类
        self.db = database(start_date='2007-01-01', end_date='2017-03-31')

    # 取计算分析师覆盖因子的原始数据
    def get_coverage_number(self):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        # 取每天，每只股票的净利润预测数据
        sql_query = "select create_date, code, count(forecast_profit) as num_fore from " \
                    "((select id, code, organ_id, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) "\
                    "group by create_date, code " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        coverage = original_data.pivot_table(index='create_date', columns='code', values='num_fore')
        # 因为发现gg数据库里的数据，每天分不同时间点有不同的数据，因此要再resample一下
        coverage = coverage.resample('d').sum().dropna(axis=0, how='all')
        # 对过去90天进行rolling求和，注意，coverage的日期索引包含非交易日
        rolling_coverage = coverage.rolling(90, min_periods=0).apply(lambda x:np.nansum(x))
        # 将日期重索引为交易日后再shift
        # 策略数据，注意shift
        rolling_coverage = rolling_coverage.reindex(self.strategy_data.stock_price.major_axis).shift(1)
        # 将其储存到raw_data中，顺便将stock code重索引为标准的stock code
        self.strategy_data.raw_data = pd.Panel({'coverage':rolling_coverage}, major_axis=
            self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

    # 计算因子值
    def get_abn_coverage(self):
        if os.path.isfile('coverage.csv'):
            self.strategy_data.raw_data = data.read_data(['coverage'], ['coverage'], shift=True)
            print('reading coverage\n')
        else:
            self.get_coverage_number()
            print('getting coverage\n')

        # 计算ln(1+coverage)得到回归的y项
        self.strategy_data.raw_data['ln_coverage'] = np.log(self.strategy_data.raw_data.ix['coverage'] + 1)
        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'], shift=True)
        turnover = (data_to_be_used.ix['Volume']/data_to_be_used.ix['FreeShares']).rolling(252).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj']/data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(252).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        # 生成调仓日
        self.generate_holding_days(holding_freq='m', start_date='2007-01-01')

        # 建立储存数据的dataframe
        abn_coverage = self.strategy_data.raw_data.ix['ln_coverage', self.holding_days, :] * np.nan
        self.reg_stats = pd.Panel(np.nan, items=['coef', 't_stats', 'rsquare'],
                        major_axis=self.holding_days, minor_axis=['int', 'lncap', 'turnover', 'momentum'])
        # 对调仓日进行循环回归
        for cursor, time in enumerate(self.holding_days):
            y = self.strategy_data.raw_data.ix['ln_coverage', time, :]
            x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], time, :]
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 1:
                continue
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            abn_coverage.ix[time] = results.resid
            self.reg_stats.ix['coef', time, :] = results.params.values
            self.reg_stats.ix['t_stats', time, :] = results.tvalues.values
            self.reg_stats.ix['rsquare', time, 0] = results.rsquared
            self.reg_stats.ix['rsquare', time, 1] = results.rsquared_adj

        abn_coverage = abn_coverage.reindex(self.strategy_data.stock_price.major_axis, method='ffill')
        self.strategy_data.factor = pd.Panel({'abn_coverage':abn_coverage}, major_axis=
                    self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

        # 对回归得到的系数取平均值
        self.table1a = self.reg_stats.mean(axis=1)

    def get_table1b(self):
        # 首先计算各种要用到的数据
        self.strategy_data.stock_price['vlty'] = self.strategy_data.stock_price.ix['daily_return'].rolling(252).std()
        # relative spread 定义不明，暂时不做
        bp = data.read_data(['bp'], shift=True)
        bp = bp.ix['bp']
        self.strategy_data.stock_price['lbm'] = np.log(1 + bp)
        roa_data = data.read_data(['TotalAssets', 'NetIncome_ttm'], shift=True)
        self.strategy_data.stock_price['roa'] = roa_data['NetIncome_ttm'] / roa_data['TotalAssets']

        base = pd.concat([self.strategy_data.raw_data.ix['coverage'], self.strategy_data.factor.ix['abn_coverage'],
                          self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum', 'vlty',
                                                            'lbm', 'roa']]], axis=0)
        self.base = pd.Panel(base.values, items=['coverage', 'abn_coverage', 'lncap', 'turnover', 'momentum',
                                                 'vlty', 'lbm', 'roa'], major_axis=base.major_axis,
                             minor_axis=base.minor_axis)
        stats = pd.Panel(np.nan, items=['obs', 'coverage', 'abn_coverage', 'lncap', 'turnover', 'momentum',
                                        'vlty', 'lbm', 'roa'], major_axis=self.holding_days, minor_axis=np.arange(10))
        # 循环调仓日，建立分位数统计量
        for cursor, time in enumerate(self.holding_days):
            curr_data = self.base.ix[:, time, :]
            # 如果abn coverage数据全是0，则继续循环
            if curr_data['abn_coverage'].isnull().all():
                continue
            group_label = pd.qcut(curr_data['abn_coverage'], 10, labels=False)
            stats.ix['obs', time, :] = curr_data['coverage'].groupby(group_label).size()
            stats.ix['coverage', time, :] = curr_data['coverage'].groupby(group_label).apply(lambda x:x.sum()/x.size)
            stats.ix[2:, time, :] = curr_data.iloc[:, 1:].groupby(group_label).mean().T.values

        # 循环结束后,对时间序列上的值取均值
        self.table1b = stats.mean(axis=1)

    # 不断加入因子回归, 看rsquare adj的路径长什么样
    def get_fig2(self):
        # 不要abn coverage因子
        y_data = self.base.ix['coverage']
        x_data = self.base.iloc[2:]
        # 储存累计r square以及最终t_stats
        self.figure2 = pd.DataFrame(np.nan, index=['r_square', 't_stats'], columns=x_data.items)
        # 有多少x维度
        dim_x = x_data.shape[0]
        # 循环递增自变量
        k=1
        while k <= dim_x:
            # 储存回归结果
            reg_results = pd.DataFrame(np.nan, index=self.holding_days,
                                       columns=['r_square']+[i for i in x_data.items])
            # 循环调仓日进行回归
            for cursor, time in enumerate(self.holding_days):
                y = y_data.ix[time, :]
                x = x_data.ix[0:k, time, :]
                x = sm.add_constant(x)
                # 如果只有小于等于1个有效数据，则返回nan序列
                if pd.concat([y, x], axis=1).dropna().shape[0] <= 1:
                    continue
                model = sm.OLS(y, x, missing='drop')
                results = model.fit()
                reg_results.ix[time, 0] = results.rsquared_adj
                reg_results.ix[time, 1:(k+1)] = results.tvalues[1:].values
            # 循环结束, 储存这轮回归的平均rsquared adj
            self.figure2.ix['r_square', k-1] = reg_results.ix[:, 0].mean()
            k += 1
        # 当结束最后一次循环的时候, 储存各回归系数的t stats
        self.figure2.ix['t_stats', :] = reg_results.ix[:, 1:].mean().values
        pass


    # 对原始数据的描述，即对论文的图表进行复制
    def data_description(self):
        self.get_abn_coverage()

        self.get_table1b()
        self.get_fig2()
        pass




if __name__ == '__main__':
    ac = analyst_coverage()
    ac.data_description()


































































