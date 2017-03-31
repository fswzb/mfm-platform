#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:41:06 2017

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

# barra base类，计算barra的风格因子以及行业因子，以此作为基础
# 注意：此类中参照barra计算的因子中，excess return暂时都没有减去risk free rate

class barra_base(object):
    """This is the class for barra base construction.
    
    foo
    """
    
    def __init__(self, *, stock_pool='all'):
        self.bb_data = strategy_data()
        self.bb_factor_return = pd.DataFrame()
        # 提示barra base的股票池
        self.bb_data.stock_pool = stock_pool
        # 提示是否为数据更新
        self.is_update = False
        
    # 建立指数加权序列
    @staticmethod
    def construct_expo_weights(halflife, length):
        exponential_lambda = 0.5 ** (1 / halflife)
        exponential_weights = (1 - exponential_lambda) * exponential_lambda ** np.arange(length-1,-1,-1)
        exponential_weights = exponential_weights/np.sum(exponential_weights)
        return exponential_weights

    # 读取在计算风格因子中需要用到的原始数据，注意：行业信息数据不在这里读取
    # 即使可以读取已算好的因子，也在这里处理，因为这样方便统一处理，不至于让代码太乱
    # 这里的标准为，读取前都检查一下是否已经存在数据，这样可以方便手动读取特定数据
    def read_original_data(self):
        # 先读取市值
        if self.bb_data.stock_price.empty:
            self.bb_data.stock_price = data.read_data(['FreeMarketValue'], ['FreeMarketValue'])
        elif 'FreeMarketValue' not in self.bb_data.stock_price.items:
            mv = data.read_data(['FreeMarketValue'], ['FreeMarketValue'])
            self.bb_data.stock_price['FreeMarketValue'] = mv.ix['FreeMarketValue']
        # 初始化无风险利率序列
        if os.path.isfile('const_data.csv'):
            self.bb_data.const_data = pd.read_csv('const_data.csv', index_col=0, parse_dates=True, encoding='GB18030')
            if 'risk_free' not in self.bb_data.const_data.columns:
                self.bb_data.const_data['risk_free'] = 0
            else:
                print('risk free rate successfully loaded')
        else:
            self.bb_data.const_data = pd.DataFrame(0, index=self.bb_data.stock_price.major_axis, columns=['risk_free'])
        # 读取价格数据
        if 'ClosePrice_adj' not in self.bb_data.stock_price.items:
            temp_closeprice = data.read_data(['ClosePrice_adj'], ['ClosePrice_adj'])
            self.bb_data.stock_price['ClosePrice_adj'] = temp_closeprice.ix['ClosePrice_adj']
        # 计算每只股票的日对数收益率
        if 'daily_return' not in self.bb_data.stock_price.items:
            self.bb_data.stock_price['daily_return'] = np.log(self.bb_data.stock_price.ix['ClosePrice_adj'].div(
                self.bb_data.stock_price.ix['ClosePrice_adj'].shift(1)))
        # 计算每只股票的日超额收益
        if 'daily_excess_return' not in self.bb_data.stock_price.items:
            self.bb_data.stock_price['daily_excess_return'] = self.bb_data.stock_price.ix['daily_return'].sub(
                self.bb_data.const_data.ix[:, 'risk_free'], axis=0)
        # 读取交易量数据
        if 'Volume' not in self.bb_data.stock_price.items:
            volume = data.read_data(['Volume'], ['Volume'])
            self.bb_data.stock_price['Volume'] = volume.ix['Volume']
        # 读取流通股数数据
        if 'FreeShares' not in self.bb_data.stock_price.items:
            shares = data.read_data(['FreeShares'], ['FreeShares'])
            self.bb_data.stock_price['FreeShares'] = shares.ix['FreeShares']
        # 读取pb
        if self.bb_data.raw_data.empty:
            self.bb_data.raw_data = data.read_data(['PB'],['PB'])
            # 一切的数据标签都以stock_price为准
            self.bb_data.raw_data = data.align_index(self.bb_data.stock_price.ix[0], 
                                                     self.bb_data.raw_data, axis = 'both')
        elif 'PB' not in self.bb_data.raw_data.items:
            pb = data.read_data(['PB'], ['PB'])
            self.bb_data.raw_data['PB'] = pb.ix['PB']
        # 读取ni_fy1, ni_fy2
        if 'NetIncome_fy1' not in self.bb_data.raw_data.items:
            NetIncome_fy1 = data.read_data(['NetIncome_fy1'], ['NetIncome_fy1'])
            self.bb_data.raw_data['NetIncome_fy1'] = NetIncome_fy1.ix['NetIncome_fy1']
        if 'NetIncome_fy2' not in self.bb_data.raw_data.items:
            NetIncome_fy2 = data.read_data(['NetIncome_fy2'], ['NetIncome_fy2'])
            self.bb_data.raw_data['NetIncome_fy2'] = NetIncome_fy2.ix['NetIncome_fy2']
        # 读取cash_earnings_ttm，现金净流入的ttm
        if 'CashEarnings_ttm' not in self.bb_data.raw_data.items:
            CashEarnings_ttm = data.read_data(['CashEarnings_ttm'], ['CashEarnings_ttm'])
            self.bb_data.raw_data['CashEarnings_ttm'] = CashEarnings_ttm.ix['CashEarnings_ttm']
        # 读取pe_ttm
        if 'PE_ttm' not in self.bb_data.raw_data.items:
            pe_ttm = data.read_data(['PE_ttm'], ['PE_ttm'])
            self.bb_data.raw_data['PE_ttm'] = pe_ttm.ix['PE_ttm']
        # 读取净利润net income ttm
        if 'NetIncome_ttm' not in self.bb_data.raw_data.items:
            ni_ttm = data.read_data(['NetIncome_ttm'], ['NetIncome_ttm'])
            self.bb_data.raw_data['NetIncome_ttm'] = ni_ttm.ix['NetIncome_ttm']
        # 读取ni ttm的2年增长率，用ni增长率代替eps增长率，因为ni增长率的数据更全
        if 'NetIncome_ttm_growth_8q' not in self.bb_data.raw_data.items:
            ni_ttm_growth_8q = data.read_data(['NetIncome_ttm_growth_8q'], ['NetIncome_ttm_growth_8q'])
            self.bb_data.raw_data['NetIncome_ttm_growth_8q'] = ni_ttm_growth_8q.ix['NetIncome_ttm_growth_8q']
        # 读取revenue ttm的2年增长率
        if 'Revenue_ttm_growth_8q' not in self.bb_data.raw_data.items:
            Revenue_ttm_growth_8q = data.read_data(['Revenue_ttm_growth_8q'], ['Revenue_ttm_growth_8q'])
            self.bb_data.raw_data['Revenue_ttm_growth_8q'] = Revenue_ttm_growth_8q.ix['Revenue_ttm_growth_8q']
        # 读取总资产和总负债，用资产负债率代替复杂的leverage因子
        if 'TotalAssets' not in self.bb_data.raw_data.items:
            TotalAssets = data.read_data(['TotalAssets'], ['TotalAssets'])
            self.bb_data.raw_data['TotalAssets'] = TotalAssets.ix['TotalAssets']
        if 'Totalliability' not in self.bb_data.raw_data.items:
            Totalliability = data.read_data(['Totalliability'], ['Totalliability'])
            self.bb_data.raw_data['Totalliability'] = Totalliability.ix['Totalliability']
        # 生成可交易及可投资数据
        self.bb_data.generate_if_tradable()
        self.bb_data.handle_stock_pool()
        # 读取完所有数据后，过滤数据
        # 注意：在之后的因子计算中，中间计算出的因子之间相互依赖的，都要再次过滤，如一个需要从另一个中算出
        # 或者回归，正交化，而且凡是由多个因子加权得到的因子，都属于这一类
        # 以及因子的计算过程中用到非此时间点的原始数据时，如在a时刻的因子值要用到在a-t时刻的原始数据
        # 需要算暴露的时候，一定要过滤uninv的数据，因为暴露是在股票池中计算的，即正交化的时候也需要过滤uninv
        # 在barra base中，事实上只有beta需要不依赖于股票池的全局计算，在beta因子计算过后，即可过滤uninv
        # 但同时注意，一旦过滤uninv，数据就不能再作为一般的因子值储存了
        self.bb_data.discard_untradable_data()

    # 计算市值对数因子，市值需要第一个计算以确保各个panel有index和column
    def get_lncap(self):
        # 如果有文件，则直接读取
        if os.path.isfile('lncap.csv') and not self.is_update:
            self.bb_data.factor = data.read_data(['lncap'], ['lncap'])
        # 没有就用市值进行计算
        else:
            self.bb_data.factor = pd.Panel({'lncap':np.log(self.bb_data.stock_price.ix['FreeMarketValue'])},
                                             major_axis=self.bb_data.stock_price.major_axis,
                                             minor_axis=self.bb_data.stock_price.minor_axis)

    # 计算beta因子
    def get_beta(self):
        if os.path.isfile('beta.csv') and not self.is_update:
            beta = data.read_data(['beta'], ['beta'])
            self.bb_data.factor['beta'] = beta.ix['beta']
        else:
            # 所有股票的日对数收益的市值加权，加权用前一交易日的市值数据进行加权
            cap_wgt_universe_return = self.bb_data.stock_price.ix['daily_excess_return'].mul(
                                       self.bb_data.stock_price.ix['FreeMarketValue'].shift(1)).div(
                                       self.bb_data.stock_price.ix['FreeMarketValue'].shift(1).sum(1), axis=0).sum(1)
            
            # 指数权重
            exponential_weights = barra_base.construct_expo_weights(63, 252)
            # 回归函数
            def reg_func(y, *, x, weights):
                # 如果y全是nan或只有一个不是nan，则直接返回nan，可自由设定阈值
                if y.notnull().sum() <= 100:
                    return pd.Series({'beta':np.nan,'hsigma':np.nan})
                x = sm.add_constant(x)
                model = sm.WLS(y, x, weights=weights, missing='drop')
                results = model.fit()
                resid = results.resid
                # 在这里提前计算hsigma----------------------------------------------------------------------
                # 求std，252个交易日，63的半衰期
                exponential_weights_h = barra_base.construct_expo_weights(63, 252)
                # 给weights加上index以索引resid
                exponential_weights_h = pd.Series(exponential_weights_h, index=y.index)
                hsigma = (resid*exponential_weights_h).std()
                # ----------------------------------------------------------------------------------------
                return pd.Series({'beta':results.params[1],'hsigma':hsigma})
            # 按照Barra的方法进行回归
            # 储存回归结果的dataframe
            temp_beta = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            temp_hsigma = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            for cursor, date in enumerate(self.bb_data.stock_price.ix['daily_excess_return'].index):
                # 至少第252期时才回归
                if cursor<=250:
                    continue
                curr_data = self.bb_data.stock_price.ix['daily_excess_return',cursor-251:cursor+1,:]
                curr_x = cap_wgt_universe_return.ix[cursor-251:cursor+1]
                temp = curr_data.apply(reg_func, x=curr_x, weights=exponential_weights)
                temp_beta.ix[cursor,:] = temp.ix['beta']
                temp_hsigma.ix[cursor,:] = temp.ix['hsigma']
                print(cursor)
                pass
            self.bb_data.factor['beta'] = temp_beta
            self.temp_hsigma = temp_hsigma
            pass

    # beta parallel
    def get_beta_parallel(self):
        if os.path.isfile('beta.csv') and not self.is_update:
            beta = data.read_data(['beta'], ['beta'])
            self.bb_data.factor['beta'] = beta.ix['beta']
        else:
            # 所有股票的日对数收益的市值加权，加权用前一交易日的市值数据进行加权
            cap_wgt_universe_return = self.bb_data.stock_price.ix['daily_excess_return'].mul(
                self.bb_data.stock_price.ix['FreeMarketValue'].shift(1)).div(
                self.bb_data.stock_price.ix['FreeMarketValue'].shift(1).sum(1), axis=0).sum(1)

            # 指数权重
            exponential_weights = barra_base.construct_expo_weights(63, 252)

            # 回归函数
            def reg_func(y, *, x, weights):
                # 如果y全是nan或只有一个不是nan，则直接返回nan，可自由设定阈值
                if y.notnull().sum() <= 100:
                    return pd.Series({'beta': np.nan, 'hsigma': np.nan})
                x = sm.add_constant(x)
                model = sm.WLS(y, x, weights=weights, missing='drop')
                results = model.fit()
                resid = results.resid
                # 在这里提前计算hsigma----------------------------------------------------------------------
                # 求std，252个交易日，63的半衰期
                exponential_weights_h = barra_base.construct_expo_weights(63, 252)
                # 给weights加上index以索引resid
                exponential_weights_h = pd.Series(exponential_weights_h, index=y.index)
                hsigma = (resid * exponential_weights_h).std()
                # ----------------------------------------------------------------------------------------
                return pd.Series({'beta': results.params[1], 'hsigma': hsigma})
            # 按照Barra的方法进行回归
            # 计算每期beta的函数
            def one_time_beta(cursor):
                curr_data = self.bb_data.stock_price.ix['daily_excess_return', cursor - 251:cursor+1, :]
                curr_x = cap_wgt_universe_return.ix[cursor - 251:cursor+1]
                temp = curr_data.apply(reg_func, x=curr_x, weights=exponential_weights)
                print(cursor)
                return temp
            import pathos.multiprocessing as mp
            if __name__ == '__main__':
                ncpus = 4
                p = mp.ProcessPool(ncpus)
                # 从252期开始
                data_size = np.arange(251, self.bb_data.stock_price.ix['daily_excess_return'].shape[0])
                chunksize = int(len(data_size)/ncpus)
                results = p.map(one_time_beta, data_size, chunksize=chunksize)
                # 储存结果
                beta = pd.concat([i.ix['beta'] for i in results], axis=1).T
                hsigma = pd.concat([i.ix['hsigma'] for i in results], axis=1).T
                # 两个数据对应的日期，为原始数据的日期减去251，因为前251期的数据并没有计算
                data_index = self.bb_data.stock_price.iloc[:, 251-self.bb_data.stock_price.shape[1]:, :].major_axis
                beta = beta.set_index(data_index)
                hsigma = hsigma.set_index(data_index)
                self.bb_data.factor['beta'] = beta
                self.temp_hsigma = hsigma.reindex(self.bb_data.stock_price.major_axis)



    # 计算momentum因子 
    def get_momentum(self):
        if os.path.isfile('momentum.csv') and not self.is_update:
            momentum = data.read_data(['momentum'], ['momentum'])
            self.bb_data.factor['momentum'] = momentum.ix['momentum']
        else:
            # 计算momentum因子
            # 首先数据有一个21天的lag
            lag_return = self.bb_data.stock_price.ix['daily_excess_return'].shift(21)
            # rolling后求sum，504个交易日，126的半衰期
            exponential_weights = barra_base.construct_expo_weights(126, 504)
            # 定义momentum的函数
            def func_mom(df, *, weights):
                iweights = pd.Series(weights, index=df.index)
                return df.mul(iweights, axis=0).sum(0)
            momentum = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            for cursor, date in enumerate(lag_return.index):
                # 至少504+21期才开始计算
                if cursor<=(502+21):
                    continue
                curr_data = lag_return.ix[cursor-503:cursor+1, :]
                temp = func_mom(curr_data, weights=exponential_weights)
                momentum.ix[cursor, :] = temp
            self.bb_data.factor['momentum'] = momentum
            pass
        
     # 计算residual volatility中的dastd
    def get_rv_dastd(self):
        if os.path.isfile('dastd.csv') and not self.is_update:
            dastd = data.read_data(['dastd'], ['dastd'])
            dastd = dastd.ix['dastd']
        else:
            # rolling后求std，252个交易日，42的半衰期
            exponential_weights = barra_base.construct_expo_weights(42, 252)
            # 定义dastd的函数
            def func_dastd(df, *, weights):
                iweights = pd.Series(weights, index=df.index)
                return df.mul(iweights, axis=0).std(0)
            dastd = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            for cursor, date in enumerate(self.bb_data.stock_price.ix['daily_excess_return'].index):
                # 至少252期才开始计算
                if cursor<=250:
                    continue
                curr_data = self.bb_data.stock_price.ix['daily_excess_return', cursor-251:cursor+1,:]
                temp = func_dastd(curr_data, weights=exponential_weights)
                dastd.ix[cursor,:] = temp
  
        self.bb_data.raw_data['dastd'] = dastd
    
    # 计算residual volatility中的cmra
    def get_rv_cmra(self):
        if os.path.isfile('cmra.csv') and not self.is_update:
            cmra = data.read_data(['cmra'], ['cmra'])
            cmra = cmra.ix['cmra']
        else:
            # 定义需要cmra的函数，这个函数计算252个交易日中的cmra
            def func_cmra(df):
                # 累计收益率
                cum_df = df.cumsum(axis=0)
                # 取每月的累计收益率
                months = np.arange(20,252,21)
                months_cum_df = cum_df.ix[months]
                z_max = months_cum_df.max(axis=0)
                z_min = months_cum_df.min(axis=0)
#                # 避免出现log函数中出现非正参数
#                z_min[z_min <= -1] = -0.9999
#                return np.log(1+z_max)-np.log(1+z_min)
                # 为避免出现z_min<=-1调整后的极端值，cmra改为z_max-z_min
                # 注意：改变后并未改变因子排序，而是将因子原本的scale变成了exp(scale)
                return z_max - z_min
            cmra = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            for cursor, date in enumerate(self.bb_data.stock_price.ix['daily_excess_return'].index):
                # 至少252期才开始计算
                if cursor <= 250:
                    continue
                curr_data = self.bb_data.stock_price.ix['daily_excess_return', cursor-251:cursor+1, :]
                temp = func_cmra(curr_data)
                cmra.ix[cursor,:] = temp
        self.bb_data.raw_data['cmra'] = cmra
    
    # 计算residual volatility中的hsigma
    def get_rv_hsigma(self):
        if os.path.isfile('hsigma.csv') and not self.is_update:
            hsigma = data.read_data(['hsigma'], ['hsigma'])
            hsigma = hsigma.ix['hsigma']
        elif hasattr(self, 'temp_hsigma'):
            hsigma = self.temp_hsigma
        else:
            print('hsigma has not been accquired, if you have rv file stored instead, ingored this message.\n')
            hsigma = np.nan
        self.bb_data.raw_data['hsigma'] = hsigma
    
    # 计算residual volatility
    def get_residual_volatility(self):
        self.get_rv_dastd()
        self.get_rv_cmra()
        self.get_rv_hsigma()
        # 过滤数据，因为之前的因子数据之后要正交化，会影响计算
        # 此处为barra base计算中第一次过滤掉uninv数据，此后的数据都不能再储存，因为依赖于stock pool
        self.bb_data.discard_uninv_data()
        if os.path.isfile('rv.csv') and not self.is_update:
            rv = data.read_data(['rv'], ['rv'])
            self.bb_data.factor['rv'] = rv.ix['rv']
        else:
            # 计算三个成分因子的暴露
            self.bb_data.raw_data['dastd_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.bb_data.raw_data.ix['dastd'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['cmra_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.bb_data.raw_data.ix['cmra'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['hsigma_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.bb_data.raw_data.ix['hsigma'], self.bb_data.stock_price.ix['FreeMarketValue'])
            
            rv = 0.74*self.bb_data.raw_data.ix['dastd_expo']+0.16*self.bb_data.raw_data.ix['cmra_expo']+ \
                                                        0.1*self.bb_data.raw_data.ix['hsigma_expo']
            # 计算rv的因子暴露，不再去极值
            y = strategy_data.get_cap_wgt_exposure(rv, self.bb_data.stock_price.ix['FreeMarketValue'], percentile=0)
            # 计算市值因子与beta因子的暴露
            x = pd.Panel({'lncap_expo':strategy_data.get_cap_wgt_exposure(self.bb_data.factor.ix['lncap'],
                                                                          self.bb_data.stock_price.ix['FreeMarketValue']),
                          'beta_expo':strategy_data.get_cap_wgt_exposure(self.bb_data.factor.ix['beta'],
                                                                         self.bb_data.stock_price.ix['FreeMarketValue'])})
            # 正交化
            new_rv = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.bb_data.stock_price.ix['FreeMarketValue']))
            # 之后会再次的计算暴露，注意再次计算暴露后，new_rv依然保有对x的正交性
            self.bb_data.factor['rv'] = new_rv
                           
    # 计算nonlinear size
    def get_nonlinear_size(self):
        if os.path.isfile('nls.csv') and not self.is_update:
            nls = data.read_data(['nls'], ['nls'])
            self.bb_data.factor['nls'] = nls.ix['nls']
        else:
            size_cube = self.bb_data.factor.ix['lncap']**3
            # 计算原始nls的暴露
            y = strategy_data.get_cap_wgt_exposure(size_cube, self.bb_data.stock_price.ix['FreeMarketValue'])
            # 计算市值因子的暴露，注意解释变量需要为一个panel
            x = pd.Panel({'lncap_expo': strategy_data.get_cap_wgt_exposure(self.bb_data.factor.ix['lncap'],
                                                                           self.bb_data.stock_price.ix['FreeMarketValue'])})
            # 对市值因子做正交化
            new_nls = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.bb_data.stock_price.ix['FreeMarketValue']))
            self.bb_data.factor['nls'] = new_nls

    # 计算pb
    def get_pb(self):
        if os.path.isfile('bp.csv') and not self.is_update:
            pb = data.read_data(['bp'], ['bp'])
            self.bb_data.factor['bp'] = pb.ix['bp']
        else:
            self.bb_data.factor['bp'] = 1/self.bb_data.raw_data.ix['PB']

    
    # 计算liquidity中的stom
    def get_liq_stom(self):
        if os.path.isfile('stom.csv') and not self.is_update:
            stom = data.read_data(['stom'], ['stom'])
            stom = stom.ix['stom']
        else:
            v2s = self.bb_data.stock_price.ix['Volume'].div(self.bb_data.stock_price.ix['FreeShares'])
            stom = v2s.rolling(21, min_periods=5).apply(lambda x:np.log(np.sum(x)))
        self.bb_data.raw_data['stom'] = stom
        # 过滤数据，因为stom会影响之后stoq，stoa的计算
        self.bb_data.discard_uninv_data()
        
    # 计算liquidity中的stoq
    def get_liq_stoq(self):
        if os.path.isfile('stoq.csv') and not self.is_update:
            stoq = data.read_data(['stoq'], ['stoq'])
            stoq = stoq.ix['stoq']
        else:
            # 定义stoq的函数
            def func_stoq(df):
                # 去过去3个月的stom
                months = np.arange(20,63,21)
                months_stom = df.ix[months]
                return np.log(np.exp(months_stom).mean(axis=0))
            stoq = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            for cursor, date in enumerate(self.bb_data.stock_price.ix['daily_excess_return'].index):
                # 至少63期才开始计算
                if cursor<=61:
                    continue
                curr_data = self.bb_data.raw_data.ix['stom', cursor-62:cursor+1,:]
                temp = func_stoq(curr_data)
                stoq.ix[cursor,:] = temp
        self.bb_data.raw_data['stoq'] = stoq

    # 计算liquidity中的stoa
    def get_liq_stoa(self):
        if os.path.isfile('stoa.csv') and not self.is_update:
            stoa = data.read_data(['stoa'], ['stoa'])
            stoa = stoa.ix['stoa']
        else:
            # 定义stoa的函数
            def func_stoa(df):
                # 去过去12个月的stom
                months = np.arange(20,252,21)
                months_stom = df.ix[months]
                return np.log(np.exp(months_stom).mean(axis=0))
            stoa = self.bb_data.stock_price.ix['daily_excess_return']*np.nan
            for cursor, date in enumerate(self.bb_data.stock_price.ix['daily_excess_return'].index):
                # 至少252期才开始计算
                if cursor<=250:
                    continue
                curr_data = self.bb_data.raw_data.ix['stom', cursor-251:cursor+1,:]
                temp = func_stoa(curr_data)
                stoa.ix[cursor,:] = temp
        self.bb_data.raw_data['stoa'] = stoa

    # 计算liquidity
    def get_liquidity(self):
        self.get_liq_stom()
        self.get_liq_stoq()
        self.get_liq_stoa()
        # 过滤数据
        self.bb_data.discard_uninv_data()
        if os.path.isfile('liquidity.csv') and not self.is_update:
            liquidity = data.read_data(['liquidity'], ['liquidity'])
            self.bb_data.factor['liquidity'] = liquidity.ix['liquidity']
        else:
            # 计算三个成分因子的暴露
            self.bb_data.raw_data['stom_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.bb_data.raw_data.ix['stom'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['stoq_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.bb_data.raw_data.ix['stoq'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['stoa_expo'] = strategy_data.get_cap_wgt_exposure( 
                    self.bb_data.raw_data.ix['stoa'], self.bb_data.stock_price.ix['FreeMarketValue'])
            
            liquidity = 0.35*self.bb_data.raw_data.ix['stom_expo']+0.35*self.bb_data.raw_data.ix['stoq_expo']+ \
                                                              0.3*self.bb_data.raw_data.ix['stoa_expo']
            # 计算liquidity的因子暴露，不再去极值
            y = strategy_data.get_cap_wgt_exposure(liquidity, self.bb_data.stock_price.ix['FreeMarketValue'], percentile=0)
            # 计算市值因子的暴露
            x = pd.Panel({'lncap_expo': strategy_data.get_cap_wgt_exposure(self.bb_data.factor.ix['lncap'],
                                                                           self.bb_data.stock_price.ix['FreeMarketValue'])})
            # 正交化
            new_liq = strategy_data.simple_orth_gs(y, x, weights = np.sqrt(self.bb_data.stock_price.ix['FreeMarketValue']))
            self.bb_data.factor['liquidity'] = new_liq

    # 计算earnings yield中的epfwd
    def get_ey_epfwd(self):
        if os.path.isfile('epfwd.csv') and not self.is_update:
            epfwd = data.read_data(['epfwd'], ['epfwd'])
            epfwd = epfwd.ix['epfwd']
        else:
            # 定义计算epfwd的函数
            def epfwd_func(fy1_data, fy2_data):
                # 获取当前的月份数
                curr_month = fy1_data.index.month
                # 获取fy1数据与fy2数据的权重，注意：财年是以4月份结束的
                # 因此5月份时，全部用fy1数据，其权重为1，fy2权重为0
                # 4月份时，fy1权重为1/12， fy2权重为11/12
                # 6月份时，fy1权重为11/12，fy2权重为1/12
                # 当前月份与5月的差距
                diff_month = curr_month-5
                fy1_weight = np.where(diff_month>=0, (12-diff_month)/12, -diff_month/12)
                # fy1_weight为一个ndarray，将它改为series
                fy1_weight = pd.Series(fy1_weight, index=fy1_data.index)
                fy2_weight = 1-fy1_weight
                return (fy1_data.mul(fy1_weight, axis=0) + fy2_data.mul(fy2_weight, axis=0))
            # 用预测的净利润数据除以市值数据得到预测的ep
            ep_fy1 = self.bb_data.raw_data.ix['NetIncome_fy1']/self.bb_data.stock_price.ix['FreeMarketValue']
            ep_fy2 = self.bb_data.raw_data.ix['NetIncome_fy2']/self.bb_data.stock_price.ix['FreeMarketValue']
            epfwd = epfwd_func(ep_fy1, ep_fy2)
        self.bb_data.raw_data['epfwd'] = epfwd
            
    # 计算earnings yield中的cetop
    def get_ey_cetop(self):
        if os.path.isfile('cetop.csv') and not self.is_update:
            cetop = data.read_data(['cetop'], ['cetop'])
            cetop = cetop.ix['cetop']
        else:
            # 用cash earnings ttm 除以市值
            cetop = self.bb_data.raw_data.ix['CashEarnings_ttm']/self.bb_data.stock_price.ix['FreeMarketValue']
        self.bb_data.raw_data['cetop'] = cetop
        
    # 计算earnings yield中的etop
    def get_ey_etop(self):
        if os.path.isfile('etop.csv') and not self.is_update:
            etop = data.read_data(['etop'], ['etop'])
            etop = etop.ix['etop']
        else:
            # 用pe_ttm的倒数来计算etop
            etop = 1/self.bb_data.raw_data.ix['PE_ttm']
        self.bb_data.raw_data['etop'] = etop

    # 计算earnings yield
    def get_earnings_yeild(self):
        self.get_ey_epfwd()
        self.get_ey_cetop()
        self.get_ey_etop()
        self.bb_data.discard_uninv_data()
        if os.path.isfile('ey.csv') and not self.is_update:
            EarningsYield = data.read_data(['ey'], ['ey'])
            self.bb_data.factor['ey'] = EarningsYield.ix['ey']
        else:
            # 计算三个成分因子的暴露
            self.bb_data.raw_data['epfwd_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['epfwd'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['cetop_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['cetop'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['etop_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['etop'], self.bb_data.stock_price.ix['FreeMarketValue'])

            EarningsYield = 0.68*self.bb_data.raw_data.ix['epfwd_expo']+0.21*self.bb_data.raw_data.ix['cetop_expo']+ \
                                0.11*self.bb_data.raw_data.ix['etop_expo']
            self.bb_data.factor['ey'] = EarningsYield

    # 计算growth中的egrlf
    def get_g_egrlf(self):
        if os.path.isfile('egrlf.csv') and not self.is_update:
            egrlf = data.read_data(['egrlf'], ['egrlf'])
            egrlf = egrlf.ix['egrlf']
        else:
            # 用ni_fy2来代替长期预测的净利润
            egrlf = (self.bb_data.raw_data.ix['NetIncome_fy2']/self.bb_data.raw_data.ix['NetIncome_ttm'])**(1/2) - 1
        self.bb_data.raw_data['egrlf'] = egrlf

    # 计算growth中的egrsf
    def get_g_egrsf(self):
        if os.path.isfile('egrsf.csv') and not self.is_update:
            egrsf = data.read_data(['egrsf'], ['egrsf'])
            egrsf = egrsf.ix['egrsf']
        else:
            # 用ni_fy1来代替短期预测净利润
            egrsf = self.bb_data.raw_data.ix['NetIncome_fy1'] / self.bb_data.raw_data.ix['NetIncome_ttm'] - 1
        self.bb_data.raw_data['egrsf'] = egrsf

    # 计算growth中的egro
    def get_g_egro(self):
        if os.path.isfile('egro.csv') and not self.is_update:
            egro = data.read_data(['egro'], ['egro'])
            egro = egro.ix['egro']
        else:
            # 用ni ttm的两年增长率代替ni ttm的5年增长率
            egro = self.bb_data.raw_data.ix['NetIncome_ttm_growth_8q']
        self.bb_data.raw_data['egro'] = egro

    # 计算growth中的sgro
    def get_g_sgro(self):
        if os.path.isfile('sgro.csv') and not self.is_update:
            sgro = data.read_data(['sgro'], ['sgro'])
            sgro = sgro.ix['sgro']
        else:
            # 用历史营业收入代替历史sales per share
            sgro = self.bb_data.raw_data.ix['Revenue_ttm_growth_8q']
        self.bb_data.raw_data['sgro'] = sgro

    # 计算growth
    def get_growth(self):
        self.get_g_egrlf()
        self.get_g_egrsf()
        self.get_g_egro()
        self.get_g_sgro()
        self.bb_data.discard_uninv_data()
        if os.path.isfile('growth.csv') and not self.is_update:
            growth = data.read_data(['growth'], ['growth'])
            self.bb_data.factor['growth'] = growth.ix['growth']
        else:
            # 计算四个成分因子的暴露
            self.bb_data.raw_data['egrlf_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['egrlf'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['egrsf_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['egrsf'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['egro_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['egro'], self.bb_data.stock_price.ix['FreeMarketValue'])
            self.bb_data.raw_data['sgro_expo'] = strategy_data.get_cap_wgt_exposure(
                self.bb_data.raw_data.ix['sgro'], self.bb_data.stock_price.ix['FreeMarketValue'])

            growth = 0.18*self.bb_data.raw_data.ix['egrlf_expo']+0.11*self.bb_data.raw_data.ix['egrsf_expo']+ \
                             0.24*self.bb_data.raw_data.ix['egro_expo']+0.47*self.bb_data.raw_data.ix['sgro_expo']
            self.bb_data.factor['growth'] = growth

    # 计算leverage
    def get_leverage(self):
        if os.path.isfile('leverage.csv') and not self.is_update:
            leverage = data.read_data(['leverage'], ['leverage'])
            self.bb_data.factor['leverage'] = leverage.ix['leverage']
        else:
            # 用简单的资产负债率计算leverage
            leverage = self.bb_data.raw_data.ix['Totalliability']/self.bb_data.raw_data.ix['TotalAssets']
            self.bb_data.factor['leverage'] = leverage

    # 计算风格因子的因子暴露
    def get_style_factor_exposure(self):
        # 给因子暴露panel加上索引
        self.bb_data.factor_expo = pd.Panel(data=None, major_axis=self.bb_data.factor.major_axis,
                                            minor_axis=self.bb_data.factor.minor_axis)
        # 循环计算暴露
        for item, df in self.bb_data.factor.iteritems():
            # 通过内部因子加总得到的因子，或已经计算过一次暴露的因子（如正交化过），不再需要去极值
            if item in ['rv', 'nls', 'liquidity', 'ey', 'growth']:
                self.bb_data.factor_expo[item] = strategy_data.get_cap_wgt_exposure(df,
                                        self.bb_data.stock_price.ix['FreeMarketValue'], percentile=0)
            else:
                self.bb_data.factor_expo[item] = strategy_data.get_cap_wgt_exposure(df,
                                        self.bb_data.stock_price.ix['FreeMarketValue'])

    # 得到行业因子的虚拟变量
    def get_industry_factor(self):
        # 读取行业信息数据
        industry = data.read_data(['Industry'],['Industry'])
        self.industry = industry.ix['Industry']
        # 对第一个日期取虚拟变量，以建立储存数据的panel
        temp_dum = pd.get_dummies(self.industry.ix[0], prefix='Industry')
        industry_dummies = pd.Panel(data=None, major_axis = temp_dum.index, minor_axis = temp_dum.columns)
        # 开始循环
        for time, ind_data in self.industry.iterrows():
            industry_dummies[time] = pd.get_dummies(ind_data, prefix='Industry')
        # 转置
        industry_dummies = industry_dummies.transpose(2,0,1)
        # 将行业因子暴露与风格因子暴露的索引对其
        industry_dummies = data.align_index(self.bb_data.factor_expo.ix[0], industry_dummies)
        # 将行业因子暴露与风格因子暴露衔接在一起
        self.bb_data.factor_expo = pd.concat([self.bb_data.factor_expo, industry_dummies])
        
    # 加入国家因子，也即回归中用到的截距项
    def add_country_factor(self):
        # 给items中的最后加上截距项，即barra里的country factor
        constant = pd.DataFrame(1, index=self.bb_data.factor_expo.major_axis,
                                columns=self.bb_data.factor_expo.minor_axis)
        constant = constant.astype(float)
        constant.name = 'country_factor'
        self.bb_data.factor_expo = pd.concat([self.bb_data.factor_expo, constant])

    # 构建barra base的所有风格因子和行业因子
    def construct_barra_base(self):
        # 读取数据，更新数据则不用读取，因为已经存在
        if not self.is_update:
            self.read_original_data()
        # 创建风格因子
        self.get_lncap()
        self.get_beta()
        self.get_momentum()
        self.get_residual_volatility()
        self.get_nonlinear_size()
        self.get_pb()
        self.get_liquidity()
        self.get_earnings_yeild()
        self.get_growth()
        self.get_leverage()
        # 计算风格因子暴露之前再过滤一次
        self.bb_data.discard_uninv_data()
        # 计算风格因子暴露
        self.get_style_factor_exposure()
        # 加入行业暴露
        self.get_industry_factor()
        # 添加国家因子
        self.add_country_factor()
        # 计算的最后，过滤数据
        self.bb_data.discard_uninv_data()

        # 如果不是更新数据且股票池为所有股票，则储存因子值数据
        # 还需要求为非外部调用的情况
        if not self.is_update and self.bb_data.stock_pool == 'all' and __name__=='__main__':
            data.write_data(self.bb_data.factor)

    # 仅计算barra base的因子暴露，主要用于对与不同股票池，可以在不重新建立新对象的情况下，算不同的因子暴露
    def just_get_factor_expo(self):
        self.bb_data.discard_uninv_data()
        self.get_style_factor_exposure()
        self.get_industry_factor()
        self.add_country_factor()
        self.bb_data.discard_uninv_data()

    # 回归计算各个基本因子的因子收益
    def get_bb_factor_return(self):
        # 初始化储存因子收益的dataframe
        self.bb_factor_return = pd.DataFrame(np.nan, index=self.bb_data.factor_expo.major_axis,
                                             columns=self.bb_data.factor_expo.items)
        # 因子暴露要用上一期的因子暴露，用来加权的市值要用上一期的市值
        lag_factor_expo = self.bb_data.factor_expo.shift(1).reindex(
                          major_axis=self.bb_data.factor_expo.major_axis)
        lag_mv = self.bb_data.stock_price.ix['FreeMarketValue'].shift(1)
        # 循环回归，计算因子收益
        for time, temp_data in self.bb_factor_return.iterrows():
            outcome = strategy_data.constrained_gls_barra_base(
                       self.bb_data.stock_price.ix['daily_return', time, :],
                       lag_factor_expo.ix[:, time, :],
                       weights = np.sqrt(lag_mv.ix[time, :]),
                       indus_ret_weights = lag_mv.ix[time, :])
            self.bb_factor_return.ix[time, :] = outcome[0]
            print(time)

    # 更新数据
    def update_barra_base_factor_data(self):
        self.is_update = True
        # 检验stock pool是否为all
        assert self.bb_data.stock_pool == 'all', print('Please make sure stock pool is all when updating factor data.\n')
        # 首先读取原始数据
        self.read_original_data()
        # 读取旧的因子数据
        old_bb_factor_names = ['lncap', 'beta', 'momentum', 'rv', 'nls', 'bp', 'liquidity', 'ey', 'growth', 'leverage']
        old_bb_factors = data.read_data(old_bb_factor_names, old_bb_factor_names)
        # 更新与否取决于原始数据和因子数据，若因子数据的时间轴早于原始数据，则进行更新
        # 这里对比的数据实际是free mv和lncap，因为barra base的计算是以这两个为基准的
        last_day = old_bb_factors.major_axis[-1]
        if last_day == self.bb_data.stock_price.major_axis[-1]:
            print('The barra base factor data have been up-to-date.\n')
            return
        # 找因子数据的最后一天在原始数据中的对应位置
        last_loc = self.bb_data.stock_price.major_axis.get_loc(last_day)
        # 将原始数据截取，截取范围从更新的第一天的（即因子数据的最后一天的下一天）前525天到最后一天
        # 更新前525天的选取是因为t时刻的bb因子值最远需要取到525天前的原始数据，在momentum因子中用到
        new_start_loc = last_loc + 1 - 525
        self.bb_data.stock_price = self.bb_data.stock_price.iloc[:, new_start_loc:, :]
        self.bb_data.raw_data = self.bb_data.raw_data.iloc[:, new_start_loc:, :]
        self.bb_data.if_tradable = self.bb_data.if_tradable.iloc[:, new_start_loc:, :]

        # 开始计算新的因子值
        self.construct_barra_base()

        # 衔接新旧因子值
        new_factor_data = pd.concat([old_bb_factors, self.bb_data.factor], axis=1)
        self.bb_data.factor = new_factor_data.groupby(new_factor_data.major_axis).first()
        # 储存因子值数据
        data.write_data(self.bb_data.factor)

        self.is_update = False

if __name__ == '__main__':
    test = barra_base()
    test.read_original_data()
    test.get_lncap()
    test.get_beta_parallel()
    pass

