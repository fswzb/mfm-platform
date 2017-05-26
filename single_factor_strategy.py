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
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix

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
        # 每个因子策略都需要用到是否可交易的数据
        self.strategy_data.generate_if_tradable(shift=True)
        # 读取市值数据以进行市值加权
        self.strategy_data.stock_price = data.read_data(['FreeMarketValue'],['FreeMarketValue'],shift = True)
        # 用来画图的pdf对象
        self.pdfs = 'default'
        
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
        pass

    # 分行业选股，跟上面的选股方式一样，只是在每个行业里选固定比例的股票
    # weight等于0为等权，等于1为直接市值加权，等于2则进行行业内与行业间的不同加权
    # inner与outter weights为0，为行业内，行业间等权，为1为行业内，行业间市值加权
    # outter weights为3，为行业间以指数权重加权（若为全市场，则改为市值加权）
    def select_stocks_within_indus(self, *, select_ratio = [0.8, 1], direction = '+', weight=0, inner_weights=1,
                                   outter_weights=1):
        # 读取行业数据：
        industry = data.read_data(['Industry'], ['Industry'])
        industry = industry['Industry']
        # 定义选股的函数
        def get_stks(factor_data, *, select_ratio=[0.8, 1], direction='+'):
            holding = pd.Series(0, index=factor_data.index)
            # 取有效的股票数
            effective_num = factor_data.dropna().size
            # 无股票可选，进行下一次循环
            if effective_num == 0:
                return holding
            # 对因子值进行排序，注意这里的秩（rank），类似于得分
            if direction is '+':
                factor_score = factor_data.rank(ascending=True)
            elif direction is '-':
                factor_score = factor_data.rank(ascending=False)
            else:
                print('Please enter ''+'' or ''-'' for direction argument')
            # 选取股票的得分范围
            lower_bound = np.floor(effective_num * select_ratio[0])
            upper_bound = np.floor(effective_num * select_ratio[1])
            # 选取股票
            selected_stocks = factor_score.ix[np.logical_and(factor_score >= lower_bound,
                                                             factor_score <= upper_bound)].index
            # 被选取的股票都将持仓调为1
            holding.ix[selected_stocks] = 1
            return holding
        # 对调仓期进行循环
        for cursor, time in self.holding_days.iteritems():
            # 当前数据
            curr_data = pd.DataFrame({'factor':self.strategy_data.factor.ix[0, time, :], 'industry':industry.ix[time]})
            # 根据行业分类选股
            curr_holding_matrix = curr_data.groupby('industry')['factor'].apply(get_stks, select_ratio=select_ratio,
                                                                                direction=direction).fillna(0)
            self.position.holding_matrix.ix[time] = curr_holding_matrix
            pass
        # 对不可交易的股票进行过滤
        if self.strategy_data.stock_pool == 'all':
            # 去除不可交易的股票
            self.filter_untradable()
        else:
            # 有股票池的情况去除不可投资的股票
            self.filter_uninv()

        # 选择加权的方式
        self.position.to_percentage()
        if weight == 1:
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
        elif weight == 2 and self.strategy_data.stock_pool == 'all':
            self.position.weighted_holding(self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :])
        elif weight == 2 and self.strategy_data.stock_pool != 'all':
            self.position.weighted_holding_indus(industry, inner_weights=self.strategy_data.stock_price.ix['FreeMarketValue',
                                           self.position.holding_matrix.index, :], outter_weights=self.strategy_data.
                                           benchmark_price.ix['Weight_'+self.strategy_data.stock_pool,
                                                              self.position.holding_matrix.index, :])
        pass

    # 用优化的方法构造纯因子组合，纯因子组合保证组合在该因子上有暴露（注意，并不一定是1），在其他因子上无暴露
    # 当优化方法中的方差矩阵为回归权重的逆矩阵时，优化方法和回归方法得到一样的权重（见Barra, efficient replication of factor returns），
    # 这时这里的结果和用回归计算的正交化提纯后的因子的因子收益一样，但是把它做成策略可以放到回测中进行回测，
    # 从而可以考虑交易成本和实际情况。注意：这里先做因子相对barra base的纯因子组合，之后可添加相对任何因子的
    # 这里先做一个简单的直接用解析解算出的组合
    def select_stocks_pure_factor_bb(self, *, bb_expo, cov_matrix='Empty', reg_weight='Empty', direction='+',
                                     regulation_lambda=1):
        # 计算因子值的暴露
        factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                                         self.strategy_data.stock_price.ix['FreeMarketValue'])
        if direction == '-':
            factor_expo = - factor_expo
        self.strategy_data.factor_expo = pd.Panel({'factor_expo':factor_expo},
                major_axis=self.strategy_data.factor.major_axis, minor_axis=self.strategy_data.factor.minor_axis)

        # 循环调仓日
        for cursor, time in self.holding_days.iteritems():
            # 当前的因子暴露向量，为n*1
            x_alpha = self.strategy_data.factor_expo.ix['factor_expo', time, :].fillna(0)
            # 当前的其他因子暴露向量，为n*(k-1)，实际就是barra base因子的暴露
            x_sigma = bb_expo.ix[:, time, :].fillna(0)

            # 有协方差矩阵，优先用协方差矩阵
            if type(cov_matrix) != str:
                inv_v = np.linalg.pinv(cov_matrix.ix[time].fillna(0))
            else:
                assert type(reg_weight) != str, 'The construction of pure factor portfolio require one of following:\n' \
                                                'Covariance matrix of factor returns (priority), OR \n' \
                                                'Regression weight when getting factor return using linear regression.\n'
                # 取当期的回归权重，每只股票的权重在对角线上
                # inv_v = np.diag(reg_weight.ix[time].fillna(0))
                curr_weight = reg_weight.ix[time]
                curr_weight = (curr_weight/curr_weight.sum()).fillna(0)
                inv_v = np.diag(curr_weight)

            # 通过优化的解析解计算权重，解析解公式见barra, Efficient Replication of Factor Returns, equation (6)
            temp_1 = np.linalg.pinv(np.dot(np.dot(x_sigma.T, inv_v), x_sigma))
            temp_2 = np.dot(np.dot(x_sigma.T, inv_v), x_alpha)
            temp_3 = x_alpha - np.dot(np.dot(x_sigma, temp_1), temp_2)
            h_star = 1/regulation_lambda * np.dot(inv_v, temp_3)

            # 加权方式只能为这一种，只是需要归一化一下
            self.position.holding_matrix.ix[time] = h_star

        self.position.to_percentage()
        pass

    # 上一种选股方法的优化解法
    def select_stocks_pure_factor(self, *, base_expo, cov_matrix='Empty', reg_weight='Empty', direction='+',
                                  benchmark_weight='Empty', is_long_only=True):
        # 计算因子值的暴露
        factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                                         self.strategy_data.stock_price.ix['FreeMarketValue'])
        if direction == '-':
            factor_expo = - factor_expo
        self.strategy_data.factor_expo = pd.Panel({'factor_expo': factor_expo},
                                                  major_axis=self.strategy_data.factor.major_axis,
                                                  minor_axis=self.strategy_data.factor.minor_axis)
        # 如果有benchmark，则计算benchmark的暴露
        if type(benchmark_weight) != str:
            benchmark_weight = (benchmark_weight.div(benchmark_weight.sum(1), axis=0)).fillna(0)
            adjusted_base_expo = strategy_data.adjust_benchmark_related_expo(base_expo,
                                    benchmark_weight, self.strategy_data.if_tradable.ix['if_tradable'])
            benchmark_base_expo = np.einsum('ijk,jk->ji', adjusted_base_expo.fillna(0), benchmark_weight.fillna(0))
            benchmark_base_expo = pd.DataFrame(benchmark_base_expo, index=base_expo.major_axis, columns=base_expo.items)

            adjusted_factor_expo = strategy_data.adjust_benchmark_related_expo(
                pd.Panel({'factor_expo':factor_expo}), benchmark_weight, self.strategy_data.if_tradable.ix['if_tradable']
            )
            adjusted_factor_expo = adjusted_factor_expo.ix['factor_expo']
            benchmark_curr_factor_expo = (adjusted_factor_expo * benchmark_weight).sum(1)
            self.strategy_data.factor_expo.ix['factor_expo'] = factor_expo.sub(benchmark_curr_factor_expo, axis=0)

        # 循环调仓日
        for cursor, time in self.holding_days.iteritems():
            curr_factor_expo = self.strategy_data.factor_expo.ix['factor_expo', time, :]
            curr_base_expo = base_expo.ix[:, time, :]

            # 有协方差矩阵，优先用协方差矩阵
            if type(cov_matrix) != str:
                curr_v = cov_matrix.ix[time]
                curr_v_diag = curr_v.diagonal()
                # 去除有nan的数据
                all_data = pd.concat([curr_v_diag, curr_factor_expo, curr_base_expo], axis=1)
                all_data = all_data.dropna()
                # 如果有效数据小于等于1，当期不选股票
                if all_data.shape[0] <= 1:
                    continue
                # 指数中选股可能会出现一个行业暴露全是0的情况，所以关于这个行业的限制条件会冗余，于是要进行剔除
                all_data = all_data.replace(0, np.nan).dropna(axis=1, how='all').fillna(0.0)
                curr_factor_expo = all_data.ix[:, 0]
                curr_v_diag = all_data.ix[:, 1]
                curr_base_expo = all_data.ix[:, 2:]
                curr_v = curr_v.reindex(index=curr_v_diag.index, columns=curr_v_diag.index)
            else:
                assert type(reg_weight) != str, 'The construction of pure factor portfolio require one of following:\n' \
                                                'Covariance matrix of factor returns (priority), OR \n' \
                                                'Regression weight when getting factor return using linear regression.\n'
                # 取当期的回归权重，每只股票的权重在对角线上
                curr_v_diag = reg_weight.ix[time]
                # 去除有nan的数据
                all_data = pd.concat([curr_v_diag, curr_factor_expo, curr_base_expo], axis=1)
                all_data = all_data.dropna()
                # 如果有效数据小于等于1，当期不选股票
                if all_data.shape[0] <= 1:
                    continue
                # 指数中选股可能会出现一个行业暴露全是0的情况，所以关于这个行业的限制条件会冗余，于是要进行剔除
                all_data = all_data.replace(0, np.nan).dropna(axis=1, how='all').fillna(0.0)
                curr_v_diag = all_data.ix[:, 0]
                curr_factor_expo = all_data.ix[:, 1]
                curr_base_expo = all_data.ix[:, 2:]
                # 将回归权重归一化
                curr_v_diag = curr_v_diag / curr_v_diag.sum()
                curr_v = np.linalg.pinv(np.diag(curr_v_diag))
                curr_v = pd.DataFrame(curr_v, index=curr_factor_expo.index, columns=curr_factor_expo.index)

            # 设置其他因子为0的限制条件，在有基准的时候，设置为基准的暴露
            if type(benchmark_weight) != str:
                expo_target = benchmark_base_expo.ix[time].reindex(index=curr_base_expo.columns)
            else:
                expo_target = pd.Series(0.0, index=curr_base_expo.columns)

            # 开始设置优化
            # P = V
            P = matrix(curr_v.as_matrix())
            # q = - (factor_expo.T)
            q = matrix(-curr_factor_expo.as_matrix().transpose())

            # 其他因为暴露为0，或等于基准的限制条件
            A = matrix(curr_base_expo.as_matrix().transpose())
            b = matrix(expo_target.as_matrix())

            solvers.options['show_progress'] = False

            # 如果只能做多，则每只股票的比例都必须大于等于0
            if is_long_only:
                long_only_constraint = pd.DataFrame(-1.0*np.eye(curr_factor_expo.size), index=curr_factor_expo.index,
                                                   columns=curr_factor_expo.index)
                long_only_target = pd.Series(0.0, index=curr_factor_expo.index)

                G = matrix(long_only_constraint.as_matrix())
                h = matrix(long_only_target.as_matrix())

                # 解优化问题
                results = solvers.qp(P=P, q=q, A=A, b=b, G=G,  h=h)
            else:
                results = solvers.qp(P=P, q=q, A=A, b=b)

            results_np = np.array(results['x']).squeeze()
            results_s = pd.Series(results_np, index=curr_factor_expo.index)
            # 重索引为所有股票代码
            results_s = results_s.reindex(self.strategy_data.stock_price.minor_axis, fill_value=0)

            # 股票持仓
            self.position.holding_matrix.ix[time] = results_s

        # 循环结束后，进行权重归一化
        self.position.to_percentage()
        pass

    # 单因子的因子收益率计算和检验，用来判断因子有效性，
    # holding_freq为回归收益的频率，默认为月，可调整为与调仓周期一样，也可不同
    # weights为用来回归的权重，默认为等权回归
    def get_factor_return(self, *, holding_freq='m', weights='default', direction='+', plot_cum=True,
                          start='default', end='default'):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算因子收益的频率
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price.\
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 如果有指定，只取start和end之间的时间计算
        if start != 'default':
            holding_days = holding_days[start:]
        if end != 'default':
            holding_days = holding_days[:end]
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

        # 输出的string
        tstats_sig_ratio = self.t_stats_series[np.abs(self.t_stats_series) >= 2].size / self.t_stats_series.size
        target_str = 'The average return of this factor: {0:.4f}%\n' \
                     'Note that the return of factor is not annualized but corresponding to the holding days interval\n' \
                     'The average t-statistics value: {1:.4f}\n' \
                     'Ratio of t_stats whose absolute value >= 2: {2:.2f}%\n'.format(
            self.factor_return_series.mean()*100, self.t_stats_series.mean(), tstats_sig_ratio*100
        )

        # 循环结束，输出结果
        print(target_str)
        with open(str(os.path.abspath('.'))+'/'+self.strategy_data.stock_pool+'/performance.txt',
                  'a', encoding='GB18030') as text_file:
            text_file.write(target_str)

        # 画图，默认画因子收益的累计收益图
        fx = plt.figure()
        ax = fx.add_subplot(1,1,1)
        zero_series = pd.Series(np.zeros(self.factor_return_series.shape), index=self.factor_return_series.index)
        if plot_cum:
            plt.plot(self.factor_return_series.cumsum()*100, 'b-')
        else:
            plt.plot(self.factor_return_series*100, 'b-')
            plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return of The Factor (%)')
        ax.set_title('The Return Series of The Factor')
        plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'FactorReturn.png', dpi=1200)
        if type(self.pdfs) != str:
            plt.savefig(self.pdfs, format='pdf')

        fx = plt.figure()
        ax = fx.add_subplot(1, 1, 1)
        plt.plot(self.t_stats_series, 'b-')
        plt.plot(zero_series, 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('T-Stats of The Factor Return')
        ax.set_title('The T-Stats Series of The Factor Return')
        plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'FactorReturnTStats.png', dpi=1200)
        if type(self.pdfs) != str:
            plt.savefig(self.pdfs, format='pdf')

    # 计算因子的IC，股票收益率是以holding_freq为频率的的收益率，默认为月
    def get_factor_ic(self, *, holding_freq='m', direction = '+', start='default', end='default'):
        # 如果没有price的数据，读入price数据，注意要shift，
        # 即本来的实现收益率应当是调仓日当天的开盘价，但这里计算调仓日前一个交易日的收盘价。
        if 'ClosePrice_adj' not in self.strategy_data.stock_price.items:
             temp_panel = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'], 
                                                            shift = True)
             self.strategy_data.stock_price['ClosePrice_adj'] = temp_panel.ix['ClosePrice_adj']
        # 计算ic的频率
        holding_days = strategy.resample_tradingdays(self.strategy_data.stock_price. \
                                                     ix['FreeMarketValue', :, 0], freq=holding_freq)
        # 如果有指定，只取start和end之间的时间计算
        if start != 'default':
            holding_days = holding_days[start:]
        if end != 'default':
            holding_days = holding_days[:end]
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
        target_str = 'The average IC of this factor: {0:.4f}\n'.format(self.ic_series.mean())
        print(target_str)
        with open(str(os.path.abspath('.'))+'/'+self.strategy_data.stock_pool+'/performance.txt',
                  'a', encoding='GB18030') as text_file:
            text_file.write(target_str)
        
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
        plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'FactorIC.png', dpi=1200)
        if type(self.pdfs) != str:
            plt.savefig(self.pdfs, format='pdf')
        
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
                
            # # 进行qcut
            # labeled_factor = pd.qcut(curr_factor_data, no_of_groups, labels = False)

            # This is a temporary solution to pandas.qcut's unique bin edge error.
            # It will be removed when pandas 0.20.0 releases, which gives an additional parameter to handle this problem
            def pct_rank_qcut(series, n):
                edges = pd.Series([float(i) / n for i in range(n + 1)])
                f = lambda x: (edges >= x).argmax()-1
                return series.rank(pct=1).apply(f).reindex(series.index)
            labeled_factor = pct_rank_qcut(curr_factor_data.dropna(), no_of_groups).reindex(curr_factor_data.index)

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
                bkt.enable_warning = False
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
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'QGroupsNetValue.png', dpi=1200)
            if type(self.pdfs) != str:
                plt.savefig(self.pdfs, format='pdf')

            # 画long-short的图
            f2 = plt.figure()
            ax2 = f2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Net Account Value')
            ax2.set_title('Net Account Value of Long-Short Portfolio of The Factor')
            plt.plot(long_series - short_series)
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'LongShortNetValue.png', dpi=1200)
            if type(self.pdfs) != str:
                plt.savefig(self.pdfs, format='pdf')

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
                bkt.enable_warning = False
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
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'QGroupsCumLog.png', dpi=1200)
            plt.savefig(self.pdfs, format='pdf')

            # 画long-short的图
            f2 = plt.figure()
            ax2 = f2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Log Return (%)')
            ax2.set_title('Cumulative Log Return of Long-Short Portfolio of The Factor')
            plt.plot((long_series - short_series) * 100)
            plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/' + 'LongShortCumLog.png', dpi=1200)
            plt.savefig(self.pdfs, format='pdf')

    # 用回归取残差的方法（即gram-schmidt正交法）取因子相对一基准的纯因子暴露
    # 之后可以用这个因子暴露当作因子进行选股，以及回归得纯因子组合收益率（主要用途），或者算ic等
    def get_pure_factor_gs_orth(self, base_expo, *, do_active_bb_pure_factor=False):
        # 计算当前因子的暴露，注意策略里的数据都已经lag过了
        factor_expo = strategy_data.get_cap_wgt_exposure(self.strategy_data.factor.iloc[0],
                                                         self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 如果计算的是相对基准的纯因子收益率
        if do_active_bb_pure_factor:
            if self.strategy_data.stock_pool == 'all':
                benchmark_weight = data.read_data(['Weight_zz500'], ['Weight_zz500'], shift=True)
            else:
                benchmark_weight = self.strategy_data.benchmark_price.ix['Weight_' + self.strategy_data.stock_pool]
            # 计算bb base的调整后暴露，以及调整后benchmark在bb base上的暴露
            adjusted_bb_expo = strategy_data.adjust_benchmark_related_expo(base_expo, benchmark_weight,
                                                                           self.strategy_data.if_tradable.ix[
                                                                               'if_tradable'])
            benchmark_bb_expo = np.einsum('ijk,jk->ji', adjusted_bb_expo.fillna(0), benchmark_weight.fillna(0))
            benchmark_bb_expo = pd.DataFrame(benchmark_bb_expo, index=base_expo.major_axis, columns=base_expo.items)
            # 计算当前因子的调整后暴露值，以及调整后benchmark在当前因子上的暴露
            adjusted_factor_expo = strategy_data.adjust_benchmark_related_expo(self.strategy_data.factor,
                                                                               benchmark_weight,
                                                                               self.strategy_data.if_tradable.ix[
                                                                                   'if_tradable'])
            adjusted_factor_expo = adjusted_factor_expo.iloc[0]
            benchmark_factor_expo = (adjusted_factor_expo * benchmark_weight).sum(1)
            # 用暴露的绝对值减去基准的暴露值，得到相对基准的超额暴露值
            base_expo = base_expo.sub(benchmark_bb_expo, axis=0)
            factor_expo = factor_expo.sub(benchmark_factor_expo, axis=0)
        # 在bb expo里去掉国家因子，去掉是为了保证有唯一解，而且去掉后残差值不变，不影响结果
        # 因为国家因子向量已经能表示成行业暴露的线性组合了
        if 'country_factor' in base_expo.items:
            base_expo_no_cf = base_expo.drop('country_factor', axis=0)
        else:
            base_expo_no_cf = base_expo
        # 利用多元线性回归进行提纯
        pure_factor_expo = strategy_data.simple_orth_gs(factor_expo, base_expo_no_cf, weights=
            np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue']), add_constant=False)
        # 将得到的纯化因子放入因子值中储存
        self.strategy_data.factor.iloc[0] = pure_factor_expo

    # 根据一个股票池进行一次完整的单因子测试的函数
    # select method为单因子测试策略的选股方式，0为按比例选股，1为分行业按比例选股
    def single_factor_test(self, *, factor='default', direction='+', bkt_obj='Empty', bb_obj='Empty',
                           pa_benchmark_weight='default', discard_factor=[], bkt_start='default', bkt_end='default',
                           stock_pool='all', select_method=0, do_pa=True, do_active_pa=False, do_bb_pure_factor=False,
                           do_active_bb_pure_factor=False, holding_freq='m', do_data_description=False):
        # 如果传入的是str，则读取同名文件，如果是dataframe，则直接传入因子
        if type(factor) == str:
            if factor != 'default':
                self.read_factor_data([factor], [factor], shift=True)
            # 检测是否已经有因子存在,因为有可能该实例化的类已经有了计算好的要测试的因子
            elif self.strategy_data.factor.shape[0]>=1:
                print('The factor has been set to be the FIRST one in strategy_data.factor\n')
            elif self.strategy_data.factor_expo.shape[0]>=1:
                self.strategy_data.factor = self.strategy_data.factor_expo
                print('The factor data has been copied from factor_expo data, and the factor will be'
                      'the FIRST one in strategy_data.factor_expo\n')
            else:
                print('Error: No factor data, please try to specify a factor!\n')
        elif self.strategy_data.factor.empty:
            self.strategy_data.factor = pd.Panel({'factor_one':factor})
        else:
            self.strategy_data.factor.iloc[0] = factor

        # 生成调仓日
        if self.holding_days.empty:
            self.generate_holding_days(holding_freq=holding_freq)
        # 初始化持仓或重置策略持仓
        if self.position.holding_matrix.empty:
            self.initialize_position(self.strategy_data.factor.ix[0, self.holding_days, :])
        else:
            self.reset_position()

        # 将策略的股票池设置为当前股票池
        self.strategy_data.stock_pool = stock_pool
        # 根据股票池生成标记
        self.strategy_data.handle_stock_pool(shift=True)
        # 除去不可交易或不可投资的数据
        # 注意，对策略数据的修改是永久性的，无法恢复，因此在使用过某个股票池的单因子测试后，对策略数据的再使用要谨慎
        if self.strategy_data.stock_pool == 'all':
            self.strategy_data.discard_untradable_data()
        else:
            self.strategy_data.discard_uninv_data()

        # 如果没有文件夹，则建立一个文件夹
        if not os.path.exists(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/'):
            os.makedirs(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/')
        # 建立画pdf的对象
        self.pdfs = PdfPages(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool + '/allfigs.pdf')

        # 如果有对原始数据的表述,则进行原始数据表述
        if do_data_description:
            self.data_description()
            print('Data description completed...\n')

        # 如果有传入的bb对象
        if bb_obj == 'Empty':
            pass
        else:
            # 将bb的股票池改为当前股票池
            bb_obj.bb_data.stock_pool = stock_pool
            # 根据股票池生成标记，注意：股票池数据不需要shift，因为这里的barrabase数据是用来事后归因的，不涉及策略构成
            bb_obj.bb_data.handle_stock_pool(shift=False)

        # 如果要做基于barra base的纯因子组合，则要对因子进行提纯
        if do_bb_pure_factor:
            # 计算因子暴露
            bb_obj.just_get_factor_expo()
            # 注意，因为这里是用bb对因子进行提纯，而不是用bb归因，因此bb需要lag一期，才不会用到未来信息
            # 否则就是用未来的bb信息来对上一期的已知的因子进行提纯，而这里因子暴露的计算lag不会影响归因时候的计算
            # 因为归因时候的计算会用没有lag的因子值和其他bb数据重新计算暴露
            lag_bb_expo = bb_obj.bb_data.factor_expo.shift(1).reindex(major_axis=bb_obj.bb_data.factor_expo.major_axis)
            ac_base = lag_bb_expo.ix[['lncap','momentum','liquidity']]
            self.get_pure_factor_gs_orth(ac_base, do_active_bb_pure_factor=do_active_bb_pure_factor)

        # 按策略进行选股
        if select_method == 0:
            # 简单分位数选股
            self.select_stocks(weight=1, direction=direction, select_ratio=[0.8, 1])
        elif select_method == 1:
            # 分行业选股
            self.select_stocks_within_indus(weight=2, direction=direction)
        elif select_method == 2 or select_method == 3:
            # 用构造纯因子组合的方法选股，2为组合自己是纯因子组合，3为组合相对基准是纯因子组合
            # 首先和计算纯因子一样，要计算bb因子的暴露
            if bb_obj.bb_data.factor_expo.empty:
                bb_obj.just_get_factor_expo()
            # 同样需要lag
            lag_bb_expo = bb_obj.bb_data.factor_expo.shift(1).reindex(major_axis=bb_obj.bb_data.factor_expo.major_axis)
            # 同样不能有country factor
            lag_bb_expo_no_cf = lag_bb_expo.drop('country_factor', axis=0)
            # # 构造纯因子组合，权重使用回归权重，即市值的根号
            # 初始化temp weight为'Empty'，即如果选股方法是2，则传入默认的benchmark weight
            temp_weight = 'Empty'
            if select_method == 2:
                # self.select_stocks_pure_factor_bb(bb_expo=lag_bb_expo_no_cf, reg_weight=np.sqrt(
                #     self.strategy_data.stock_price.ix['FreeMarketValue']), direction=direction)
                self.select_stocks_pure_factor(base_expo=lag_bb_expo_no_cf, reg_weight=np.sqrt(
                    self.strategy_data.stock_price.ix['FreeMarketValue']), direction=direction,
                                               benchmark_weight=temp_weight, is_long_only=False)
            if select_method == 3 and self.strategy_data.stock_pool == 'all':
                temp_weight = data.read_data(['Weight_zz500'], ['Weight_zz500'], shift=True)
                temp_weight = temp_weight['Weight_zz500']
            elif select_method == 3 and self.strategy_data.stock_pool != 'all':
                # 注意股票池为非全市场时，基准的权重数据已经shift过了
                temp_weight = self.strategy_data.benchmark_price.ix['Weight_'+self.strategy_data.stock_pool]
            if select_method == 3:
                self.select_stocks_pure_factor(base_expo=lag_bb_expo_no_cf, reg_weight=np.sqrt(
                    self.strategy_data.stock_price.ix['FreeMarketValue']), direction=direction,
                    benchmark_weight=temp_weight, is_long_only=True)

        # 如果有外来的backtest对象，则使用这个backtest对象，如果没有，则需要自己建立，同时传入最新持仓
        if bkt_obj == 'Empty':
            bkt_obj = backtest(self.position, bkt_start=bkt_start, bkt_end=bkt_end)
        else:
            bkt_obj.reset_bkt_position(self.position)
        # 将回测的基准改为当前的股票池，若为all，则用默认的基准值
        if stock_pool != 'all':
            bkt_obj.reset_bkt_benchmark(['ClosePrice_adj_' + stock_pool])

        # 回测、画图、归因
        bkt_obj.execute_backtest()
        bkt_obj.get_performance(foldername=stock_pool, pdfs=self.pdfs)

        # 如果要进行归因的话
        if do_pa:
            # 如果指定了要做超额收益的归因，且有股票池，则用相对基准的持仓来归因
            # 而股票池为全市场时的超额归因默认基准为中证500
            if do_active_pa and self.strategy_data.stock_pool != 'all':
                # 注意：策略里的strategy_data里的数据都是shift过后的，而进行归因的数据和回测一样，不能用shift数据，要用当天最新数据
                pa_benchmark_weight = data.read_data(['Weight_'+self.strategy_data.stock_pool],
                                                     ['Weight_'+self.strategy_data.stock_pool])
                pa_benchmark_weight = pa_benchmark_weight['Weight_'+self.strategy_data.stock_pool]
            elif do_active_pa and self.strategy_data.stock_pool == 'all':
                temp_weight = data.read_data(['Weight_zz500'], ['Weight_zz500'])
                pa_benchmark_weight = temp_weight['Weight_zz500']
            # 注意bb obj进行了一份深拷贝，这是因为在业绩归因的计算中，会根据不同的股票池丢弃数据，导致数据不全，因此不能传引用
            bkt_obj.get_performance_attribution(outside_bb=bb_obj, benchmark_weight=pa_benchmark_weight,
                                                discard_factor=discard_factor, show_warning=False,
                                                foldername=stock_pool, pdfs=self.pdfs, is_real_world=True,
                                                real_world_type=1)

        # 画单因子组合收益率
        self.get_factor_return(weights=np.sqrt(self.strategy_data.stock_price.ix['FreeMarketValue']),
                               holding_freq='d', direction=direction, start=bkt_start, end=bkt_end)
        # 画ic的走势图
        self.get_factor_ic(direction=direction, holding_freq='m', start=bkt_start, end=bkt_end)
        # 画分位数图和long short图
        self.plot_qgroup(bkt_obj, 5, direction=direction, value=1, weight=1)

        self.pdfs.close()







                
        
        
        
    
            
            
            
                
                
            
            
            
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            