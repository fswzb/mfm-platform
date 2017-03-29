#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:06:35 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
from cvxopt import solvers, matrix

from data import data

# 数据类，所有数据均为pd.Panel, major_axis为时间，minor_axis为股票代码，items为数据名称

# 多因子策略数据类
class strategy_data(data):
    """ This is the multi_factor strategy data class.
    
    stock_price (pd.Panel): price data of stocks
    benchmark_price (pd.Panel): price data of benchmarks
    raw_data (pd.Panel): original data get from market or financial report, or intermediate data
                         which is used for factor calculation, note the difference between stock_price
                         data and raw_data
    factor (pd.Panel): final factors calculated which is used during process of stock selection
    factor_expo(pd.Pnael): factor exposure after standardization
    stock_pool(pd.DataFrame): stock pool to select stocks from
    """
    def __init__(self):
        data.__init__(self)
        self.factor = pd.Panel()
        self.factor_expo = pd.Panel()
        # 股票池，即策略选取的股票池，或各因子数据计算时用到的股票池
        # 目前对股票池的处理方法是将其归为不可交易，用discard_untradable_data来将股票池外的数据设为nan
        self.stock_pool = 'all'

    # 新建一个dataframe储存股票是否在股票池内，再建一个dataframe和if_tradable取交集
    def handle_stock_pool(self, *, stock_pool='all', shift=False):
        self.stock_pool = stock_pool
        # 如果未设置股票池
        if self.stock_pool == 'all':
            self.if_tradable['if_inpool'] = True
        # 设置了股票池，若已存在benchmark中的weight，则直接使用
        elif 'Weight_'+stock_pool in self.benchmark_price.items:
            self.if_tradable['if_inpool'] = self.benchmark_price.ix['Weight_'+stock_pool]>0
        # 若不在，则读取weight数据，文件名即为stock_pool
        else:
            temp_weights = data.read_data(['Weight_'+stock_pool],['Weight_'+stock_pool])
            self.benchmark_price['Weight_'+stock_pool] = temp_weights['Weight_'+stock_pool]
            self.if_tradable['if_inpool'] = self.benchmark_price.ix['Weight_'+stock_pool]>0

        if shift:
            self.if_tradable['if_inpool'] = self.if_tradable['if_inpool'].shift(1)

        # 若还没有if_tradable，报错
        assert 'if_tradable' in self.if_tradable.items, 'Please generate if_tradable first!'

        # 新建一个if_inv，表明在股票池中，且可以交易
        # 在if_tradable中为true，且在if_inpool中为true，才可投资，即在if_inv中为true
        self.if_tradable.ix['if_inv'] = np.logical_and(self.if_tradable.ix['if_tradable'], self.if_tradable.ix['if_inpool'])

    # 对数据进行winsorization
    @staticmethod
    def winsorization(raw_data, *, percentile = 0.03):
        """ Winsorize the data.
        
        raw_data (pd.DataFrame): data you'd like to winsorize
        percentile: percentile on which data will be winsorized.
        """
        temp = raw_data
        lower_q = raw_data.quantile(percentile, 1)[:, np.newaxis]
        upper_q = raw_data.quantile(1-percentile, 1)[:, np.newaxis]
        raw_data = np.where(np.greater(raw_data, lower_q), raw_data, lower_q)
        raw_data = np.where(np.less(raw_data, upper_q), raw_data, upper_q)
        # np.greater, np.less遇到nan时都返回false，因此要将nan的数据改为nan
        raw_data = np.where(temp.isnull(), np.nan, raw_data)
        return pd.DataFrame(raw_data, index=temp.index, columns=temp.columns)
            
    # 对数据进行zscore
    @staticmethod
    def zscore(raw_data):
        """ Get z-score of a series of data.
        
        raw_data (pd.DataFrame): data you'd like to get z-score from
        """
        mu = raw_data.mean(1)
        sigma = raw_data.std(1)
        raw_data = raw_data.sub(mu, axis=0).div(sigma, axis=0)
        return raw_data

    # 对数据进行zscore，均值用市值进行加权，标准差还是简单加权
    @staticmethod
    def cap_wgt_zscore(raw_data, mv):
        cap_wgt_mu = raw_data.mul(mv).div(mv.sum(1), axis=0).sum(1)
        sigma = raw_data.std(1)
        raw_data = raw_data.sub(cap_wgt_mu, axis=0).div(sigma, axis=0)
        return raw_data

    # 对数据进行rescale，将尾部分布进行压缩，方法参考eue3
    # 将大于3 sigma的数据部分压缩到一个限制范围内，默认为3.5
    # 注意输入数据为z score后的数据
    @staticmethod
    def compress_tail_data(raw_data, *, limit = 3.5):
        # 首先计算数据的均值，进行平移，使得新数据均值为0
        # 这是因为按照barra的zscore方法做出来的数据均值并不是0，标准差一定是1
        mu = raw_data.mean(1)
        centered = raw_data.sub(mu, axis=0)
        # 按照eue3的方法进行compress
        data_max = centered.max(axis=1)
        data_min = centered.min(axis=1)
        s_plus = np.maximum(np.zeros_like(data_max), np.minimum(np.ones_like(data_max), (limit-3)/(data_max-3)))
        s_minus = np.maximum(np.zeros_like(data_min), np.minimum(np.ones_like(data_min), (3-limit)/(data_min+3)))
        centered = np.where(np.less(centered, 3), centered, (centered-3).mul(s_plus, axis=0)+3)
        centered = pd.DataFrame(centered, index=raw_data.index, columns=raw_data.columns)
        centered = np.where(np.greater(centered, -3), centered, (centered+3).mul(s_minus, axis=0)-3)
        centered = pd.DataFrame(centered, index=raw_data.index, columns=raw_data.columns)
        compressed = centered.add(mu, axis=0)
        return compressed

    # 计算因子暴露，简单加权
    @staticmethod
    def get_exposure(factor, *, percentile = 0.03, compress = True, limit = 3.5):
        temp_data = strategy_data.winsorization(factor, percentile = percentile)
        # 如有需要，对尾部数据进行压缩
        if compress:
            # 先标准化
            temp_data = strategy_data.zscore(temp_data)
            # 进行压缩
            temp_data = strategy_data.compress_tail_data(temp_data, limit=limit)
        # 进行标准化
        final_data = strategy_data.zscore(temp_data)
        return final_data
    
    # 计算市值加权的因子暴露
    @staticmethod
    def get_cap_wgt_exposure(factor, mv, *, percentile = 0.03, compress = True, limit = 3.5):
        temp_data = strategy_data.winsorization(factor, percentile = percentile)
        # 如有需要，对尾部数据进行压缩
        if compress:
            # 先标准化
            temp_data = strategy_data.cap_wgt_zscore(temp_data, mv)
            # 进行压缩
            temp_data = strategy_data.compress_tail_data(temp_data, limit=limit)
        # 进行标准化
        final_data = strategy_data.cap_wgt_zscore(temp_data, mv)
        return final_data
    
    
    # 检查在某一时间，某只股票是否处于可交易状态
    def check_if_tradable(self, time, stock):
        return self.if_tradable.ix['if_tradable', time, stock]  
    
    # 将strategy_data中的所有数据，在不可交易的时候，都设为nan，
    # 在策略中：因为在shift之后，if_tradable是一个选股时的已知信息，
    # 直接利用这个信息，过滤掉调仓日之前不可交易的股票，使得选股策略更加真实有效
    # 需注意，如果在策略中使用，一定要在if_tradable数据shift之后再使用，否则会用到未来信息
    # 如果只是单纯的计算数据（如计算因子），则不需要shift if_tradable数据，因为当天的数据是用当天所有已知信息计算后储存下来的
    def discard_untradable_data(self):
        # 如果没有可交易标记的数据，则什么数据也不丢弃
        if self.if_tradable.ix['if_tradable'].empty:
            return
        
        # 股票价格行情数据
        if not self.stock_price.empty:
            for item, df in self.stock_price.iteritems():
                self.stock_price.ix[item] = self.stock_price.ix[item].where(
                                             self.if_tradable.ix['if_tradable'], np.nan)
        
        # benchmark数据
        if not self.benchmark_price.empty:
            for item, df in self.benchmark_price.iteritems():
                self.benchmark_price.ix[item] = self.benchmark_price.ix[item].where(
                                                 self.if_tradable.ix['if_tradable'], np.nan)
        
        # 原始数据                                          
        if not self.raw_data.empty:
            for item, df in self.raw_data.iteritems():
                self.raw_data.ix[item] = self.raw_data.ix[item].where(
                                          self.if_tradable.ix['if_tradable'], np.nan)
                                               
        # 因子数据
        if not self.factor.empty:
            for item, df in self.factor.iteritems():
                self.factor.ix[item] = self.factor.ix[item].where(
                                        self.if_tradable.ix['if_tradable'], np.nan)
        
        # 因子暴露数据        
        if not self.factor_expo.empty:
            for item, df in self.factor_expo.iteritems():
                self.factor_expo.ix[item] = self.factor_expo.ix[item].where(
                                             self.if_tradable.ix['if_tradable'], np.nan)

    # 与discard_untradable_data一样，只是这里丢弃掉不可投资的数据
    def discard_uninv_data(self):
        # 如果没有可交易标记的数据，则什么数据也不丢弃
        if self.if_tradable.ix['if_inv'].empty:
            return

        # 股票价格行情数据
        if not self.stock_price.empty:
            for item, df in self.stock_price.iteritems():
                self.stock_price.ix[item] = self.stock_price.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # benchmark数据
        if not self.benchmark_price.empty:
            for item, df in self.benchmark_price.iteritems():
                self.benchmark_price.ix[item] = self.benchmark_price.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # 原始数据
        if not self.raw_data.empty:
            for item, df in self.raw_data.iteritems():
                self.raw_data.ix[item] = self.raw_data.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # 因子数据
        if not self.factor.empty:
            for item, df in self.factor.iteritems():
                self.factor.ix[item] = self.factor.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        # 因子暴露数据
        if not self.factor_expo.empty:
            for item, df in self.factor_expo.iteritems():
                self.factor_expo.ix[item] = self.factor_expo.ix[item].where(
                    self.if_tradable.ix['if_inv'], np.nan)

        
    # 对数据进行回归取残差提纯，即gram-schmidt正交化
    @staticmethod
    def simple_orth_gs(obj, base, *, weights = 'default'):
        # 定义回归函数
        def reg_func(y, x, *, weights=1):
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y,x], axis=1).dropna().shape[0] <= 1:
                return pd.Series(np.nan, index=y.index)
            model = sm.WLS(y, x, weights=weights, missing='drop')
            results = model.fit()
            resid = results.resid
            return resid.reindex(y.index)
        new_obj = obj*np.nan
        if weights is 'default':
            for cursor, date in enumerate(obj.index):
                new_obj.ix[cursor] = reg_func(obj.ix[cursor], base.ix[:,cursor,:])
        else:
            for cursor, date in enumerate(obj.index):
                new_obj.ix[cursor] = reg_func(obj.ix[cursor], base.ix[:,cursor,:], weights=weights.ix[cursor])
            # 如果提纯为加权的回归，则默认提纯是为了之后这个残差和base进行加权回归时相互正交
            # 即：实际为残差和加权（加根号权重）后的base因子正交，那么在之后进行加权回归的时候，会再一次的进行加权
            # 为了避免残差因子在那个时候连加两次权，这里必须进行调整，即：除以根号权重
            # 注意除以根号权重是因为最小二乘回归的权重实际为在因子上乘以根号权重
            new_obj = new_obj.div(np.sqrt(weights))
        return new_obj
            
    # 用因子暴露数据，回归权重，进行barra模型的回归
    # 用二次规划问题求解此线性回归问题
    # 目前，基于barra的业绩归因、barra基础因子内部回归都可以用这个线性回归模型，暂不支持新增因子
    @staticmethod
    def constrained_gls_barra_base(asset_return, bb, *, weights='default',  indus_ret_weights = 'default'):
        """Solving constrained gls problem using quadratic programming.
        
        asset_return: return of asset universe.
        bb: barra base factor exposures, including style factors and industrial factors
        weights: weights of gls, usually the sqrt of mv, default means equal weight
        indus_ret_weights: weights that put on the constraints of industry factors returns, usually as the \
                           market value, default means equal weight
        """
        if weights is 'default':
            weights = pd.Series(1, index=asset_return.index)
        if indus_ret_weights is 'default':
            indus_ret_weights = pd.Series(1, index=asset_return.index)
            
        # 设置权重
        # 回归的权重需要开根号
        sqrt_w = np.sqrt(weights)
        y = asset_return.mul(sqrt_w)
        # bb中股票为index，因子名字为columns
        x = bb.mul(sqrt_w, axis=0)
        
        # 只要有na，就drop掉这只股票
        yx = pd.concat([y,x], axis=1)
        yx = yx.dropna()
        # 如果只有小于等于1个有效数据，返回nan序列
        if yx.shape[0] <= 1:
            return np.empty(x.shape[1])*np.nan
        y = yx.ix[:, 0]
        x = yx.ix[:, 1:]
        
        # 开始设置优化
        # P = X.T dot X
        P = matrix(np.dot(x.as_matrix().transpose(), x.as_matrix()))
        # q = - (X.T dot Y)
        q = matrix(-np.dot(x.as_matrix().transpose(), y.as_matrix()))
        
        # 设置行业因子收益限制条件，行业因子暴露为x中的第 11 列到第 38 列
        # 行业因子的限制权重，循环在行业因子中求和
        final_weight = pd.Series(np.arange(28)*0)
        for cursor in range(10,38):
            final_weight.ix[cursor-10] = (indus_ret_weights*x.ix[:, cursor]).sum()
        final_weight = final_weight/(final_weight.sum())
        # 设置行业因子收益的加权求和限制为0
        indus_cons = pd.Series(np.arange(39)*0)
        # 系数的第11到38项设置为行业因子收益的权重
        indus_cons.ix[10:37] = final_weight.values
        # 设置限制条件
        A = matrix(indus_cons.as_matrix(), (1, indus_cons.size))
        b = matrix(0.0)
        
        # 隐藏优化器输出
        solvers.options['show_progress'] = False
        # 解优化问题
        results = solvers.qp(P=P, q=q, A=A, b=b)
        # 将数据类型改为(n,)的ndarray
        results_np = np.array(results['x']).squeeze()

        # 计算残差
        residuals = y - x.dot(results_np)
        
        return [results_np, residuals]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    