#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:51:44 2017

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os
import statsmodels.api as sm
import copy
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix
from pandas.tools.plotting import table
from pandas.stats.fama_macbeth import fama_macbeth

from single_factor_strategy import single_factor_strategy
from database import database
from data import data
from strategy_data import strategy_data
from strategy import strategy

# 分析师预测覆盖因子的单因子策略

class analyst_coverage(single_factor_strategy):
    """Analyst coverage single factor strategy class.
    
    foo
    """
    def __init__(self):
        single_factor_strategy.__init__(self)
        # 该策略用于取数据的database类
        self.db = database(start_date='2007-01-01', end_date='2017-04-27')

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
        # 策略数据，注意shift
        rolling_coverage = rolling_coverage.reindex(self.strategy_data.stock_price.major_axis).shift(1)
        # 将其储存到raw_data中，顺便将stock code重索引为标准的stock code
        self.strategy_data.raw_data = pd.Panel({'coverage':rolling_coverage}, major_axis=
            self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

    # 定义在grouped数据(对同一股票的过去一段时间的分析师预测数据)里, 取最近一次预测的最近一个财年的预测值的函数
    @staticmethod
    def get_recent_report_recent_fy(x):
        # 删除重复报告, 只得到每个机构的每个分析师对每个财年做出的最近的那个报告
        recent_report = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'],
                                          keep='last')
        # 对这个报告的机构,分析师进行分组,取里面的第一个预测的年份
        # 要对这个年份进行排序, 取最大的那一个, 因为有可能出现, 过去一段时间(如90天), 机构预测的第一个财年不一样的情况
        # 这种多出现于4月份这种发财报的时候, 排序后取最大的那一个年份, 就是最新报告中的最近财年.
        # 注意:这里的假设是, 分析师不会跳着发布预测, 即不会出现2016年没发布2017年的预测, 直接发布2018年的
        recent_fy = recent_report.groupby(['organ_id', 'author'])['Time_year']. \
            apply(lambda x:x.dropna().iloc[0] if x.dropna().size>0 else np.nan).sort_values(ascending=True)
        # 如果当前股票的预测财年都是nan, 则返回nan, 这种情况很少见, 否则则返回第一个值,即最大值
        if recent_fy.dropna().size == 0:
            return np.nan
        else:
            return recent_fy.iloc[0]

    # 定义计算ni的std与均值的函数
    @staticmethod
    def ni_std(x):
        # 首先去除重复项
        unique_x = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'], keep='last')[
            ['Time_year', 'forecast_profit']]
        # 将每一只股票的唯一预测数据, 按照机构和作者进行分组,并取最近的那一年
        # recent_forecast = unique_x.groupby(['organ_id', 'author'])['forecast_profit'].nth(0)
        recent_fy = analyst_coverage.get_recent_report_recent_fy(x)
        # 取算出的财年的数据
        recent_forecast = unique_x['forecast_profit'].where(unique_x['Time_year'] == recent_fy, np.nan)
        # 计算std
        recent_std = recent_forecast.std()
        # 必须有3个以上的独立分析师数据, 否则数据为nan
        if recent_forecast.dropna().shape[0] >= 3:
            return recent_std
        else:
            return np.nan

    @staticmethod
    def ni_mean(x):
        # 首先去除重复项
        unique_x = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'], keep='last')
        # 将每一只股票的唯一预测数据, 按照机构和作者进行分组,并取最近的那一年
        # recent_forecast = unique_x.groupby(['organ_id', 'author'])['forecast_profit'].nth(0)
        recent_fy = analyst_coverage.get_recent_report_recent_fy(x)
        recent_forecast = unique_x['forecast_profit'].where(unique_x['Time_year'] == recent_fy, np.nan)
        # 计算mean
        recent_mean = recent_forecast.mean()
        # 必须有3个以上的独立分析师数据, 否则数据为nan
        if recent_forecast.dropna().shape[0] >= 3:
            return recent_mean
        else:
            return np.nan


    @staticmethod
    def ni_count(x):
        unique_x = x.drop_duplicates(subset=['organ_id', 'author', 'Time_year'], keep='last')
        recent_fy = analyst_coverage.get_recent_report_recent_fy(x)
        recent_forecast = unique_x['forecast_profit'].where(unique_x['Time_year'] == recent_fy, np.nan)
        # 计算count
        recent_count = recent_forecast.count()
        return recent_count

    @staticmethod
    def simple_count(x):
        unique_x = x.drop_duplicates(subset=['organ_id', 'author'], keep='last')
        recent_count = unique_x.shape[0]
        return recent_count

    # 计算滚动期内的唯一分析师分析数据
    def get_unique_coverage_number(self, *, rolling_days=90):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        # 先将时间期内的所有数据都取出来
        sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
                    "((select id, code, organ_id, author, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        # 先构造一个pivot table,主要目的是为了取时间
        date_mark = original_data.pivot_table(index='create_date', columns='code', values='forecast_profit')
        # 因为数据有每天不同时点的数据,因此要resample
        date_mark = date_mark.resample('d').mean().dropna(axis=0, how='all')
        # 建立储存数据的dataframe
        coverage = date_mark * np.nan
        # # 建立储存disp与ep的dataframe, disp为有效公司的下一年的预测ni的std/price
        # # ep为有效公司的下一年的预测ni/price, 有效公司为rolling days内至少有3个独立预测值的公司
        # coverage_disp = coverage * np.nan
        # coverage_ep = coverage * np.nan

        # 根据得到的时间索引进行循环
        for cursor, time in enumerate(date_mark.index):
            # 取最近的90天
            end_time = time
            start_time = end_time - pd.DateOffset(days=rolling_days-1)
            # 满足最近90天条件的项
            condition = np.logical_and(original_data['create_date'] >= start_time,
                                       original_data['create_date'] <= end_time)
            recent_data = original_data[condition]
            # 对股票分组
            grouped = recent_data.groupby('code')
            # 分组汇总, 若机构id,作者,预测年份都一样,则删除重复的项,然后再汇总一共有多少预测ni的报告(ni值非nan)
            # curr_coverage = grouped.apply(lambda x:x.drop_duplicates(subset=['organ_id', 'author',
            #                               'Time_year'])['forecast_profit'].count())


            if cursor>=100:
                pass
            curr_coverage = grouped.apply(analyst_coverage.simple_count)
            # coverage_std = grouped.apply(analyst_coverage.ni_std)
            # coverage_mean = grouped.apply(analyst_coverage.ni_mean)

            # 储存此数据
            coverage.ix[time, :] = curr_coverage
            # coverage_disp.ix[time, :] = coverage_std
            # coverage_ep.ix[time, :] = coverage_mean

            print(time)
        pass


        # 策略数据需要shift, 先shift再重索引为交易日,这样星期一早上能知道上周末的信息,而不是上周5的信息
        coverage = coverage.shift(1)
        # 将其储存到raw_data中,对交易日和股票代码的重索引均在这里完成
        self.strategy_data.raw_data = pd.Panel({'coverage':coverage}, major_axis=
            self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

        # # 读取市值数据
        # mv = data.read_data(['FreeMarketValue'], shift=False)
        # mv = mv['FreeMarketValue']
        # # 重索引
        # coverage_disp = coverage_disp.reindex(index=mv.index, columns=mv.columns)
        # coverage_ep = coverage_ep.reindex(index=mv.index, columns=mv.columns)
        # coverage_disp = coverage_disp/mv
        # coverage_ep = coverage_ep/mv
        # # 储存数据
        # self.strategy_data.raw_data['coverage_disp'] = coverage_disp.shift(1)
        # self.strategy_data.raw_data['coverage_ep'] = coverage_ep.shift(1)
        pass

    # 计算滚动期内的唯一分析师分析数据
    def get_unique_coverage_number_parallel(self, *, rolling_days=90):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        # 先将时间期内的所有数据都取出来
        sql_query = "select create_date, code, organ_id, author, Time_year, forecast_profit from " \
                    "((select id, code, organ_id, author, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select report_search_id as id, Time_year, forecast_profit from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.id) " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        # 先构造一个pivot table,主要目的是为了取时间
        date_mark = original_data.pivot_table(index='create_date', columns='code', values='forecast_profit')
        # 因为数据有每天不同时点的数据,因此要resample
        date_mark = date_mark.resample('d').mean().dropna(axis=0, how='all')
        # 建立储存数据的dataframe
        coverage = date_mark * np.nan
        # # 建立储存disp与ep的dataframe, disp为有效公司的下一年的预测ni的std/price
        # # ep为有效公司的下一年的预测ni/price, 有效公司为rolling days内至少有3个独立预测值的公司
        # coverage_disp = coverage * np.nan
        # coverage_ep = coverage * np.nan

        # 计算每期的coverage的函数
        def one_time_coverage(cursor):
            # 得到对应位置的时间索引,取为截至时间
            end_time = date_mark.index[cursor]
            start_time = end_time - pd.DateOffset(days=rolling_days - 1)
            # 满足最近90天条件的项
            condition = np.logical_and(original_data['create_date'] >= start_time,
                                       original_data['create_date'] <= end_time)
            recent_data = original_data[condition]
            # 对股票分组
            grouped = recent_data.groupby('code')
            # 分组汇总, 若机构id,作者,预测年份都一样,则删除重复的项,然后再汇总一共有多少预测ni的报告(ni值非nan)
            # curr_coverage = grouped.apply(lambda x: x.drop_duplicates(subset=['organ_id', 'author',
            #                               'Time_year'])['forecast_profit'].count())

            curr_coverage = grouped.apply(analyst_coverage.simple_count)
            # curr_coverage = grouped.apply(analyst_coverage.ni_count)
            # coverage_std = grouped.apply(analyst_coverage.ni_std)
            # coverage_mean = grouped.apply(analyst_coverage.ni_mean)
            # df = pd.DataFrame({'cov':curr_coverage, 'std':coverage_std, 'mean':coverage_mean})
            df = pd.DataFrame({'cov':curr_coverage})
            print(end_time)
            return df
        # 进行并行计算
        import pathos.multiprocessing as mp
        if __name__ == '__main__':
            ncpus = 16
            p = mp.ProcessPool(ncpus)
            data_size = np.arange(date_mark.shape[0])
            chunksize=int(len(data_size)/ncpus)
            results = p.map(one_time_coverage, data_size, chunksize=chunksize)
            temp_coverage = pd.concat([i['cov'] for i in results], axis=1).T
            # temp_disp = pd.concat([i['std'] for i in results], axis=1).T
            # temp_ep = pd.concat([i['mean'] for i in results], axis=1).T
            coverage[:] = temp_coverage.values
            # coverage_disp[:] = temp_disp.values
            # coverage_ep[:] = temp_ep.values
        pass

        # # 策略数据需要shift, 先shift再重索引为交易日,这样星期一早上能知道上周末的信息,而不是上周5的信息
        # coverage = coverage.shift(1)
        # 将其储存到raw_data中,对交易日和股票代码的重索引均在这里完成
        self.strategy_data.raw_data = pd.Panel({'coverage': coverage}, major_axis=
        self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)

        # # 读取市值数据
        # mv = data.read_data(['FreeMarketValue'], shift=False)
        # mv = mv['FreeMarketValue']
        # # 重索引
        # coverage_disp = coverage_disp.reindex(index=mv.index, columns=mv.columns)
        # coverage_ep = coverage_ep.reindex(index=mv.index, columns=mv.columns)
        # coverage_disp = coverage_disp / mv
        # coverage_ep = coverage_ep / mv

        # # 策略数据需要shift
        # coverage_disp = coverage_disp.shift(1)
        # coverage_ep = coverage_ep.shift(1)
        # 储存数据
        # self.strategy_data.raw_data['coverage_disp'] = coverage_disp
        # self.strategy_data.raw_data['coverage_ep'] = coverage_ep

        # data.write_data(self.strategy_data.raw_data, file_name=['unique_coverage', 'coverage_disp', 'coverage_ep'])
        data.write_data(self.strategy_data.raw_data, file_name=['unique_coverage'])

    # 计算因子值
    def get_abn_coverage(self):
        if os.path.isfile('unique_coverage.csv'):
            self.strategy_data.raw_data = data.read_data(['unique_coverage'], ['coverage'], shift=True)
            print('reading coverage\n')
        else:
            self.get_unique_coverage_number_parallel()
            print('getting coverage\n')

        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.strategy_data.raw_data.ix['coverage'] = self.strategy_data.raw_data.ix['coverage'].fillna(0)

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

        # 过滤数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()

        # 计算暴露
        for item in ['lncap', 'turnover', 'momentum']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

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
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 3:
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
        self.strategy_data.stock_price.ix['abn_coverage'] = strategy_data.get_exposure(abn_coverage,
                                                                           percentile=0, compress=False)

    # 利用泊松回归来计算abn coverage
    def get_abn_coverage_poisson(self):
        if os.path.isfile('unique_coverage.csv'):
            self.strategy_data.raw_data = data.read_data(['unique_coverage'], ['coverage'], shift=True)
            print('reading coverage\n')
        else:
            self.get_unique_coverage_number_parallel()
            print('getting coverage\n')

        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.strategy_data.raw_data.ix['coverage'] = self.strategy_data.raw_data.ix['coverage'].fillna(0)

        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'], shift=True)
        turnover = (data_to_be_used.ix['Volume'] / data_to_be_used.ix['FreeShares']).rolling(252).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj'] / data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(252).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        # 过滤数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()

        # 计算暴露
        for item in ['lncap', 'turnover', 'momentum']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        # 生成调仓日
        self.generate_holding_days(holding_freq='m', start_date='2007-01-01')
        # 建立储存数据的dataframe
        abn_coverage = self.strategy_data.raw_data.ix['coverage', self.holding_days, :] * np.nan
        self.reg_stats = pd.Panel(np.nan, items=['coef', 't_stats', 'rsquare'],
                                  major_axis=self.holding_days, minor_axis=['int', 'lncap', 'turnover', 'momentum'])

        from statsmodels.discrete.discrete_model import Poisson
        # 对调仓日进行循环回归
        for cursor, time in enumerate(self.holding_days):
            y = self.strategy_data.raw_data.ix['coverage', time, :]
            x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], time, :]
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 3:
                continue
            P = Poisson(y, x, missing='drop')
            results = P.fit(full_output=True)
            abn_coverage.ix[time] = results.resid

        abn_coverage = abn_coverage.reindex(self.strategy_data.stock_price.major_axis, method='ffill')
        self.strategy_data.factor = pd.Panel({'abn_coverage': abn_coverage}, major_axis=
        self.strategy_data.stock_price.major_axis, minor_axis=self.strategy_data.stock_price.minor_axis)
        self.strategy_data.stock_price.ix['abn_coverage'] = strategy_data.get_exposure(abn_coverage,
                                                                                       percentile=0, compress=False)


    # 定义进行fama-macbeth回归的函数, 因为论文中用到了大量的fm回归
    @staticmethod
    def fama_macbeth(y, x, *, nw_lags=0, intercept=True):
        """
        
        :param y: pd.DataFrame
        :param x: pd.Panel
        :param nw_lags: Newey-West adjustment lags
        :return: coefficents, t statitics, rsquared, rsquared adj 
        """

        # 堆叠y和x
        stacked_y = y.stack(dropna=False)
        stacked_x = x.to_frame(filter_observations=False)

        # 移除nan的项
        valid = pd.concat([stacked_y, stacked_x], axis=1).notnull().all(1)
        valid_stacked_y = stacked_y[valid]
        valid_stacked_x = stacked_x[valid]

        if nw_lags == 0:
            results_fm = fama_macbeth(y=valid_stacked_y, x=valid_stacked_x, intercept=intercept)
        else:
            results_fm = fama_macbeth(y=valid_stacked_y, x=valid_stacked_x, intercept=intercept,
                                      nw_lags_beta=nw_lags)

        r2 = results_fm._ols_result.r2.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
        r2_adj = results_fm._ols_result.r2_adj.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()

        return results_fm.mean_beta, results_fm.t_stat, r2, r2_adj

    def get_table1a(self):
        if os.path.isfile('unique_coverage.csv'):
            self.strategy_data.raw_data = data.read_data(['unique_coverage'], ['coverage'], shift=True)
            print('reading coverage\n')
        else:
            self.get_unique_coverage_number_parallel()
            print('getting coverage\n')

        # 将覆盖原始数据填上0, 之后记得要过滤数据
        self.strategy_data.raw_data.ix['coverage'] = self.strategy_data.raw_data.ix['coverage'].fillna(0)

        # 计算ln(1+coverage)得到回归的y项
        self.strategy_data.raw_data['ln_coverage'] = np.log(self.strategy_data.raw_data.ix['coverage'] + 1)
        # 计算lncap
        self.strategy_data.stock_price['lncap'] = np.log(self.strategy_data.stock_price.ix['FreeMarketValue'])
        # 计算turnover和momentum
        data_to_be_used = data.read_data(['Volume', 'FreeShares', 'ClosePrice_adj'], shift=True)
        turnover = (data_to_be_used.ix['Volume'] / data_to_be_used.ix['FreeShares']).rolling(252).sum()
        daily_return = np.log(data_to_be_used.ix['ClosePrice_adj'] / data_to_be_used.ix['ClosePrice_adj'].shift(1))
        momentum = daily_return.rolling(252).sum()
        self.strategy_data.stock_price['daily_return'] = daily_return
        self.strategy_data.stock_price['turnover'] = turnover
        self.strategy_data.stock_price['momentum'] = momentum

        # 过滤数据
        self.strategy_data.handle_stock_pool(shift=True)
        self.strategy_data.discard_uninv_data()

        # 计算暴露
        for item in ['lncap', 'turnover', 'momentum']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        # 建立储存数据的dataframe
        abn_coverage = self.strategy_data.raw_data.ix['ln_coverage', self.holding_days, :] * np.nan
        self.reg_stats = pd.Panel(np.nan, items=['coef', 't_stats', 'rsquare'],
                                  major_axis=self.holding_days, minor_axis=['int', 'lncap', 'turnover', 'momentum'])
        from statsmodels.discrete.discrete_model import Poisson
        # 对调仓日进行循环回归
        for cursor, time in enumerate(self.holding_days):
            y = self.strategy_data.raw_data.ix['ln_coverage', time, :]
            x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], time, :]
            x = sm.add_constant(x)
            # 如果只有小于等于1个有效数据，则返回nan序列
            if pd.concat([y, x], axis=1).dropna().shape[0] <= 3:
                continue
            # model = sm.OLS(y, x, missing='drop')
            # results = model.fit()
            P = Poisson(y, x, missing='drop')
            results = P.fit(full_output=True)
            abn_coverage.ix[time] = results.resid
            self.reg_stats.ix['coef', time, :] = results.params.values
            self.reg_stats.ix['t_stats', time, :] = results.tvalues.values
            # self.reg_stats.ix['rsquare', time, 0] = results.rsquared
            # self.reg_stats.ix['rsquare', time, 1] = results.rsquared_adj

        abn_coverage = abn_coverage.reindex(self.strategy_data.stock_price.major_axis, method='ffill')
        # 再次对abn coverage计算暴露, 但是不再winsorize
        self.strategy_data.stock_price['abn_coverage'] = strategy_data.get_exposure(abn_coverage,
                                                                                    percentile=0, compress=False)

        # 应当根据月份,对数据进行fm回归
        y_fm = self.strategy_data.raw_data.ix['ln_coverage', self.holding_days, :]
        x_fm = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum'], self.holding_days, :]
        # 进行fm回归
        coef, t_stat, r2, r2_adj = analyst_coverage.fama_macbeth(y_fm, x_fm)
        self.table1a = pd.DataFrame({'coef':coef, 't_stat':t_stat})
        self.table1a['r_square'] = np.nan
        self.table1a.ix[0, 'r_square'] = r2
        self.table1a.ix[1, 'r_square'] = r2_adj

        # 用csv储存结果
        self.table1a.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                     '/' + 'Table1a.csv', na_rep='N/A', encoding='GB18030')

        # # 对比用clustered se算出的t stats
        # stacked_y = self.strategy_data.raw_data.ix['ln_coverage'].stack(dropna=False)
        # stacked_x = self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum']].to_frame(filter_observations=False)
        # stacked_x = sm.add_constant(stacked_x)
        # valid = pd.concat([stacked_y, stacked_x], axis=1).notnull().all(1)
        # stacked_y = stacked_y[valid]
        # stacked_x = stacked_x[valid]
        # groups_stock = stacked_y.index.get_level_values(1).values
        # model = sm.OLS(stacked_y, stacked_x)
        # # results_cluster = model.fit(cov_type='cluster', cov_kwds={'groups':groups_stock})
        # results_cluster = model.fit()
        # self.table1a_cluster = self.table1a * np.nan
        # self.table1a_cluster['coef'] = results_cluster.params.values
        # self.table1a_cluster['t_stats'] = results_cluster.tvalues.values
        # self.table1a_cluster.ix[0, 'rsquare'] = results_cluster.rsquared
        # self.table1a_cluster.ix[1, 'rsquare'] = results_cluster.rsquared_adj
        #
        # # 储存结果
        # self.table1a_cluster.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #                     '/' + 'Table1a_cluster.csv', na_rep='N/A', encoding='GB18030')

        # 首先计算各种要用到的数据
        self.strategy_data.stock_price['vlty'] = self.strategy_data.stock_price.ix['daily_return'].rolling(252).std()
        # relative spread 定义不明，暂时不做
        bp = data.read_data(['bp'], shift=True)
        bp = bp.ix['bp']
        self.strategy_data.stock_price['lbm'] = np.log(1 + bp)
        roa_data = data.read_data(['TotalAssets', 'NetIncome_ttm'], shift=True)
        self.strategy_data.stock_price['roa'] = roa_data['NetIncome_ttm'] / roa_data['TotalAssets']

        # 读取净利润
        ni_ttm = data.read_data(['NetIncome_ttm'])
        self.strategy_data.raw_data['ni_ttm'] = ni_ttm.ix['NetIncome_ttm']

        # 过滤数据
        self.strategy_data.discard_uninv_data()

        # 计算因子暴露
        for item in ['vlty', 'lbm', 'roa']:
            self.strategy_data.stock_price.ix[item] = strategy_data.get_exposure(
                self.strategy_data.stock_price.ix[item])

        base = pd.concat([self.strategy_data.raw_data.ix['coverage'], self.strategy_data.stock_price.ix['abn_coverage'],
                          self.strategy_data.stock_price.ix[['lncap', 'turnover', 'momentum', 'vlty',
                                                             'lbm', 'roa']]], axis=0)
        self.base = pd.Panel(base.values, items=['coverage', 'abn_coverage', 'lncap', 'turnover', 'momentum',
                                                 'vlty', 'lbm', 'roa'], major_axis=base.major_axis,
                             minor_axis=base.minor_axis)

    def get_table1b(self):

        stats = pd.Panel(np.nan, items=['obs', 'coverage', 'abn_coverage', 'lncap', 'turnover', 'momentum',
                                        'vlty', 'lbm', 'roa'], major_axis=self.holding_days, minor_axis=np.arange(10))
        # 循环调仓日，建立分位数统计量
        for cursor, time in enumerate(self.holding_days):
            curr_data = self.base.ix[:, time, :]
            # 如果abn coverage数据全是0，则继续循环
            if curr_data['abn_coverage'].isnull().all():
                continue
            #
            group_label = pd.qcut(curr_data['abn_coverage'], 10, labels=False)
            stats.ix['obs', time, :] = curr_data['coverage'].groupby(group_label).size()
            stats.ix['coverage', time, :] = curr_data['coverage'].groupby(group_label).apply(lambda x:x.sum()/x.size)
            stats.ix[2:, time, :] = curr_data.iloc[:, 1:].groupby(group_label).mean().T.values

            if stats.ix['roa', time, 2] >=40:
                pass

        # 循环结束后,对时间序列上的值取均值
        # self.table1b = stats.mean(axis=1)
        self.table1b = stats.median(axis=1)
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table1b, loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table1b.png', dpi=1200)
        self.table1b.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'Table1b.csv', na_rep='N/A', encoding='GB18030')

    # 不断加入因子回归, 看rsquare adj的路径长什么样
    def get_fig2(self):
        # 从fig2开始, coverage数据全部用的是lncoverage, 因此将coverage改为lncoverage
        self.base.ix['coverage'] = np.log(self.base.ix['coverage'] + 1)
        # 不要abn coverage因子
        y_data = self.base.ix['coverage']
        x_data = self.base.iloc[2:]
        # 储存累计r square以及最终t_stats
        self.figure2 = pd.DataFrame(np.nan, index=['r_square_adj', 't_stats'], columns=x_data.items)
        # 有多少x维度
        dim_x = x_data.shape[0]
        # 循环递增自变量
        k=1
        while k <= dim_x:
            # 储存回归结果
            reg_results = pd.DataFrame(np.nan, index=self.holding_days,
                                       columns=['r_square']+[i for i in x_data.items])
            # # 循环调仓日进行回归
            # for cursor, time in enumerate(self.holding_days):
            #     y = y_data.ix[time, :]
            #     x = x_data.ix[0:k, time, :]
            #     x = sm.add_constant(x)
            #     # 如果只有小于等于1个有效数据，则返回nan序列
            #     if pd.concat([y, x], axis=1).dropna().shape[0] <= k:
            #         continue
            #     model = sm.OLS(y, x, missing='drop')
            #     results = model.fit()
            #     reg_results.ix[time, 0] = results.rsquared_adj
            #     reg_results.ix[time, 1:(k+1)] = results.tvalues[1:].values
            # # 循环结束, 储存这轮回归的平均rsquared adj
            # self.figure2.ix['r_square', k-1] = reg_results.ix[:, 0].mean()

            # 在每次循环中使用fm回归
            y = y_data.ix[self.holding_days, :]
            x = x_data.ix[0:k, self.holding_days, :]
            coef, t_stat, r2, r2_adj = analyst_coverage.fama_macbeth(y, x)
            self.figure2.ix['r_square_adj', k-1] = r2_adj
            k += 1
        # 当结束最后一次循环的时候, 储存各回归系数的t stats
        # self.figure2.ix['t_stats', :] = reg_results.ix[:, 1:].replace(np.inf, np.nan).replace(-np.inf, np.nan).mean().values
        self.figure2.ix['t_stats', :] = t_stat
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.figure2, loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Figure2.png', dpi=1200)
        self.figure2.to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'Figure2.csv', na_rep='N/A', encoding='GB18030')
        pass

    # 做标准化后, 看abn cov对之后的财务数据的预测能力, 这里选取净利润, 而且是在控制其他因子的情况下
    def get_table3(self):

        # 取用来预测的解释变量
        raw_x_data = self.base.ix[['abn_coverage', 'lncap', 'momentum', 'vlty', 'lbm']]
        # 被解释变量为净利润 或净利润增长率
        raw_ya_data = self.strategy_data.raw_data.ix['ni_ttm']
        # raw_yb_data = np.log(self.strategy_data.raw_data.ix['ni_ttm']/self.strategy_data.raw_data.ix['ni_ttm'].shift(63))
        raw_yb_data = strategy_data.get_ni_growth(self.strategy_data.raw_data.ix['ni_ttm'], lag=63)
        # 标准化
        ya_data = strategy_data.get_exposure(raw_ya_data)
        yb_data = strategy_data.get_exposure(raw_yb_data)
        x_data = raw_x_data * np.nan
        for cursor, item in enumerate(raw_x_data.items):
            x_data[item] = strategy_data.get_exposure(raw_x_data.ix[item])
        # 储存数据
        self.table3a = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=[i for i in raw_x_data.items]+\
                                ['r_square'], minor_axis=['q+1', 'q+2', 'q+3', 'q+4'])
        self.table3b = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=[i for i in raw_x_data.items]+\
                                ['r_square'], minor_axis=['q+1', 'q+2', 'q+3', 'q+4'])
        # 循环季度长度,从下一个季度的预测,预测之后第4个季度
        for next_q in np.arange(4):
            # 解释变量会被移动
            curr_ya_data = ya_data.shift(-63 * (next_q + 1))
            curr_yb_data = yb_data.shift(-63 * (next_q + 1))
            # # 接下来建立数据储存, 以及循环回归
            # reg_results_a = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=self.holding_days,
            #                          minor_axis=['abn_coverage', 'lncap', 'momentum', 'vlty', 'lbm', 'r_square'])
            # reg_results_b = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=self.holding_days,
            #                          minor_axis=['abn_coverage', 'lncap', 'momentum', 'vlty', 'lbm', 'r_square'])
            # for cursor, time in enumerate(self.holding_days):
            #     ya = curr_ya_data.ix[time, :]
            #     yb = curr_yb_data.ix[time, :]
            #     x = x_data.ix[:, time, :]
            #     x = sm.add_constant(x)
            #     # 对于2个回归,只有在有一个以上有效数据的情况下才回归
            #     if pd.concat([ya, x], axis=1).dropna().shape[0] > 5:
            #         modela = sm.OLS(ya, x, missing='drop')
            #         resultsa = modela.fit()
            #         reg_results_a.ix['coef', time, 0:5] = resultsa.params[1:].values
            #         reg_results_a.ix['t_stats', time, 0:5] = resultsa.tvalues[1:].values
            #         reg_results_a.ix['coef', time, 5] = resultsa.rsquared
            #         reg_results_a.ix['t_stats', time, 5] = resultsa.rsquared_adj
            #     if pd.concat([yb, x], axis=1).dropna().shape[0] > 5:
            #         modelb = sm.OLS(yb, x, missing='drop')
            #         resultsb = modelb.fit()
            #         reg_results_b.ix['coef', time, 0:5] = resultsb.params[1:].values
            #         reg_results_b.ix['t_stats', time, 0:5] = resultsb.tvalues[1:].values
            #         reg_results_b.ix['coef', time, 5] = resultsb.rsquared
            #         reg_results_b.ix['t_stats', time, 5] = resultsb.rsquared_adj

            # 进行fm回归
            ya = curr_ya_data.ix[self.holding_days, :]
            yb = curr_yb_data.ix[self.holding_days, :]
            x = x_data.ix[:, self.holding_days, :]
            coef_a, t_a, r2_a, r2_adj_a = analyst_coverage.fama_macbeth(ya, x)
            coef_b, t_b, r2_b, r2_adj_b = analyst_coverage.fama_macbeth(yb, x)
            self.table3a.ix['coef', :, next_q] = coef_a
            self.table3a.ix['t_stats', :, next_q] = t_a
            self.table3a.ix['coef', -1, next_q] = r2_a
            self.table3a.ix['t_stats', -1, next_q] = r2_adj_a

            self.table3b.ix['coef', :, next_q] = coef_b
            self.table3b.ix['t_stats', :, next_q] = t_b
            self.table3b.ix['coef', -1, next_q] = r2_b
            self.table3b.ix['t_stats', -1, next_q] = r2_adj_b

            # # 循环结束,计算回归结果平均数并储存
            # self.table3a.ix['coef', :, next_q] = reg_results_a.ix['coef', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            # self.table3a.ix['t_stats', :, next_q] = reg_results_a.ix['t_stats', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            # self.table3b.ix['coef', :, next_q] = reg_results_b.ix['coef', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
            # self.table3b.ix['t_stats', :, next_q] = reg_results_b.ix['t_stats', :, :].\
            #     replace(np.inf, np.nan).replace(-np.inf, np.nan).mean()
        pass
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3a.ix['coef'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3a_coef.png', dpi=1200)
        self.table3a.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                            '/' + 'Table3a_coef.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3a.ix['t_stats'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3a_t_stats.png', dpi=1200)
        self.table3a.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                       '/' + 'Table3a_t_stats.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3b.ix['coef'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3b_coef.png', dpi=1200)
        self.table3b.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                       '/' + 'Table3b_coef.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table3b.ix['t_stats'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table3b_t_stats.png', dpi=1200)
        self.table3b.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                          '/' + 'Table3b_t_stats.csv', na_rep='N/A', encoding='GB18030')

    # 用一众因子对收益进行逐步回归,其实就是算各因子的回归纯因子收益
    def get_table6(self):
        # 首先算月收益,将收益按调仓月分配到各个组中取聚合求和
        term_label = pd.cut(self.strategy_data.stock_price.major_axis.map(lambda x:x.value),
                            bins=self.holding_days.map(lambda x:x.value), labels=False)
        term_label[np.isnan(term_label)] = -1.0
        self.monthly_return = self.strategy_data.stock_price.ix['daily_return'].groupby(term_label).sum()
        # 因为是用之后的收益来回归,因此相当于预测了收益
        monthly_return = self.monthly_return.shift(-1)
        ## 选取解释变量,将base中丢掉abn coverage即可
        # 用其他回归后, 用coverage不如用abn coverage, 因此丢弃coverage
        raw_x_data = self.base.drop('coverage')
        # 计算暴露
        y_data = pd.DataFrame(monthly_return.values, index=self.holding_days, columns=monthly_return.columns)
        x_data = raw_x_data * np.nan
        x_dim = x_data.shape[0]
        for cursor, item in enumerate(raw_x_data.items):
            x_data[item] = strategy_data.get_exposure(raw_x_data.ix[item])
        # 初始化储存数据的矩阵
        self.table6 = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=[i for i in x_data.items] + \
                               ['intercept', 'r_square'], minor_axis=np.arange(x_dim))
        # 开始进行循环,每一步多添加一个因子
        k = 1
        while k <= x_dim:
            # # 建立储存数据的panel
            # reg_results = pd.Panel(np.nan, items=['coef', 't_stats'], major_axis=self.holding_days,
            #                        minor_axis=self.table6.major_axis)
            # # 循环进行回归
            # for cursor, time in enumerate(self.holding_days):
            #     y = y_data.iloc[cursor, :]
            #     x = x_data.ix[:k, time, :]
            #     x = sm.add_constant(x)
            #     # 对于2个回归,只有在有一个以上有效数据的情况下才回归
            #     if pd.concat([y, x], axis=1).dropna().shape[0] <= k:
            #         continue
            #     model = sm.OLS(y, x, missing='drop')
            #     results = model.fit()
            #     # 储存结果
            #     reg_results.ix['coef', time, :k] = results.params[1:k+1]
            #     reg_results.ix['coef', time, -2] = results.params[0]
            #     reg_results.ix['coef', time, -1] = results.rsquared
            #     reg_results.ix['t_stats', time, :k] = results.tvalues[1:k+1]
            #     reg_results.ix['t_stats', time, -2] = results.tvalues[0]
            #     reg_results.ix['t_stats', time, -1] = results.rsquared_adj
            # # 循环结束,求平均值储存
            # self.table6.ix[:, :, k-1] = reg_results.replace(np.inf, np.nan).replace(-np.inf, np.nan).mean(axis=1)

            # 进行fm回归
            y = y_data
            x = x_data.ix[:k, self.holding_days, :]
            coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x, nw_lags=10)
            self.table6.ix['coef', :k, k-1] = coef.values[:k]
            self.table6.ix['coef', 'intercept', k-1] = coef.values[-1]
            self.table6.ix['t_stats', :k, k-1] = t_stats.values[:k]
            self.table6.ix['t_stats', 'intercept', k-1] = t_stats.values[-1]
            self.table6.ix['coef', 'r_square', k-1] = r2
            self.table6.ix['t_stats', 'r_square', k-1] = r2_adj

            k += 1
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table6.ix['coef'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table6_coef.png', dpi=1200)
        self.table6.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                          '/' + 'Table6_coef.csv', na_rep='N/A', encoding='GB18030')
        # # 画表
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.set_frame_on(False)
        # table(ax, self.table6.ix['t_stats'], loc='best')
        # plt.savefig(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
        #             '/' + 'Table6_t_stats.png', dpi=1200)
        self.table6.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                          '/' + 'Table6_t_stats.csv', na_rep='N/A', encoding='GB18030')
        pass

    # 画论文中的table7, 研究abn coverage因子与分析师预测值本身的std与mean的关系
    def get_table7(self):
        # 读取disp和ep的数据
        disp_ep = data.read_data(['coverage_disp', 'coverage_ep'], shift=True)
        disp = disp_ep['coverage_disp']
        ep = disp_ep['coverage_ep']
        # abn coverage数据
        abn_coverage = self.base['abn_coverage']

        def pct_rank_qcut(x, *, n):
            if x.dropna().size <= 3:
                return pd.Series(np.nan, index=x.index)
            q_labels = pd.qcut(x, q=n, labels=['low', 'mid', 'high'])
            return q_labels

        # 将disp, ep, abn coverage分为3个分位点
        disp_tercile = disp.apply(pct_rank_qcut, axis=1, n=3)
        ep_tercile = ep.apply(pct_rank_qcut, axis=1, n=3)
        abn_coverage_tercile = abn_coverage.apply(pct_rank_qcut, axis=1, n=3)
        # 循环取得dummy变量
        disp_dummies = pd.Panel(data=None, major_axis=disp.columns, minor_axis=['disp_low', 'disp_mid', 'disp_high'])
        ep_dummies = pd.Panel(data=None, major_axis=disp.columns, minor_axis=['ep_low', 'ep_mid', 'ep_high'])
        abn_coverage_dummies = pd.Panel(data=None, major_axis=disp.columns, minor_axis=['abn_coverage_low',
                                        'abn_coverage_mid', 'abn_coverage_high'])
        for cursor, time in enumerate(self.holding_days):
            if disp_tercile.ix[time, :].dropna().size >= 3:
                disp_dummies[time] = pd.get_dummies(disp_tercile.ix[time, :], prefix='disp')
            if ep_tercile.ix[time, :].dropna().size >= 3:
                ep_dummies[time] = pd.get_dummies(ep_tercile.ix[time, :], prefix='ep')
            if abn_coverage_tercile.ix[time, :].dropna().size >= 3:
                abn_coverage_dummies[time] = pd.get_dummies(abn_coverage_tercile.ix[time, :], prefix='abn_coverage')

        # 循环结束后进行转置
        disp_dummies = disp_dummies.transpose(2, 0, 1)
        ep_dummies = ep_dummies.transpose(2, 0, 1)
        abn_coverage_dummies = abn_coverage_dummies.transpose(2, 0, 1)
        # 将所有的dummy变量链接成一个大的panel, 从中选取解释变量,并首先进行数据过滤
        dummy_base = pd.concat([abn_coverage_dummies, disp_dummies, ep_dummies], axis=0)
        for item, df in dummy_base.iteritems():
            dummy_base[item] = df.where(self.strategy_data.if_tradable['if_inv'], np.nan)

        # 储存回归结果的表
        self.table7 = pd.Panel(np.nan, items=['coef' 't_stats'], major_axis=['high ATOT', 'high ATOT & low signal',
                               'high ATOT & mid signal', 'mid ATOT', 'mid ATOT & low signal', 'mid ATOT & high signal',
                               'low ATOT', 'low ATOT & mid signal', 'low ATOT & high signal', 'ATOT', 'disp', 'ep',
                               'r_square'], minor_axis = np.arange(4))

        # 回归的被解释变量
        # 月度收益,但是注意是预测未来收益,因此shift了-1
        y = self.monthly_return.shift(-1)
        y = pd.DataFrame(y.values, index=self.holding_days, columns=y.columns)

        # 第一次回归的回归变量为atot的3个分位数
        x1 = pd.Panel({'high ATOT':dummy_base['abn_coverage_high'], 'mid ATOT':dummy_base['abn_coverage_mid'],
                       'low ATOT':dummy_base['abn_coverage_low']})
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x1, nw_lags=10, intercept=False)
        self.table7.ix['coef', ['high ATOT', 'mid ATOT', 'low ATOT'], 0] = coef
        self.table7.ix['t_stats', ['high ATOT', 'mid ATOT', 'low ATOT'], 0] = t_stats
        self.table7.ix['coef', 'r_square', 0] = r2
        self.table7.ix['t_stats', 'r_square', 0] = r2_adj

        # 计算交叉回归项的x
        def get_intersect_x(signal):
            hl = dummy_base['abn_coverage_high']*dummy_base[signal+'low']
            hm = dummy_base['abn_coverage_high']*dummy_base[signal+'mid']
            ml = dummy_base['abn_coverage_mid'] * dummy_base[signal + 'low']
            mh = dummy_base['abn_coverage_mid']*dummy_base[signal+'high']
            lm = dummy_base['abn_coverage_low']*dummy_base[signal+'mid']
            lh = dummy_base['abn_coverage_low']*dummy_base[signal+'high']\

            intersect_base = pd.Panel({'high ATOT':dummy_base['abn_coverage_high'], 'high ATOT & low signal':hl,
                                       'high ATOT & mid signal':hm, 'mid ATOT':dummy_base['abn_coverage_mid'],
                                       'mid ATOT & low signal':ml, 'mid ATOT & high signal':mh,
                                       'low ATOT':dummy_base['abn_coverage_low'], 'low ATOT & mid signal':lm,
                                       'low ATOT & high signal':lh})
            return intersect_base

        # 第二次回归为使用disp的交叉项
        x2 = get_intersect_x('disp_')
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x2, nw_lags=10, intercept=False)
        self.table7.ix['coef', 'high ATOT':'low ATOT & high signal', 1] = coef
        self.table7.ix['t_stats', 'high ATOT':'low ATOT & high signal', 1] = t_stats
        self.table7.ix['coef', 'r_square', 1] = r2
        self.table7.ix['t_stats', 'r_square', 1] = r2_adj

        # 第三次回归为使用ep的交叉项
        x3 = get_intersect_x('ep_')
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x3, nw_lags=10, intercept=False)
        self.table7.ix['coef', 'high ATOT':'low ATOT & high signal', 2] = coef
        self.table7.ix['t_stats', 'high ATOT':'low ATOT & high signal', 2] = t_stats
        self.table7.ix['coef', 'r_square', 2] = r2
        self.table7.ix['t_stats', 'r_square', 2] = r2_adj

        # 第三次回归使用abn coverage, disp的暴露, ep的暴露回归
        # 过滤数据
        disp = disp.where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        ep = ep.where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        # 计算暴露
        disp_expo = strategy_data.get_exposure(disp)
        ep_expo = strategy_data.get_exposure(ep)
        x4 = pd.Panel({'ATOT':abn_coverage, 'disp':disp_expo, 'ep':ep_expo})
        coef, t_stats, r2, r2_adj = analyst_coverage.fama_macbeth(y, x4, nw_lags=10, intercept=False)
        self.table7.ix['coef', 'ATOT':'ep', 3] = coef
        self.table7.ix['t_stats', 'ATOT':'ep', 3] = t_stats
        self.table7.ix['coef', 'r_square', 3] = r2
        self.table7.ix['t_stats', 'r_square', 3] = r2_adj
        pass

        # 储存数据
        self.table7.ix['coef'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                      '/' + 'Table7_coef.csv', na_rep='N/A', encoding='GB18030')
        self.table7.ix['t_stats'].to_csv(str(os.path.abspath('.')) + '/' + self.strategy_data.stock_pool +
                                         '/' + 'Table7_t_stats.csv', na_rep='N/A', encoding='GB18030')

    # 根据原始的coverage值, 画面板数据的kde值
    def draw_kde(self):
        unique_coverage = self.strategy_data.raw_data.ix['coverage', self.holding_days, :]
        stacked_uc = unique_coverage.stack(dropna=True)

        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        plt.hist(stacked_uc.values)
        plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) + '/kde.png', dpi=1200)

        uc_old = data.read_data(['unique_coverage 90'], ['coverage_old'])
        uc_old = uc_old['coverage_old'].fillna(0).where(self.strategy_data.if_tradable.ix['if_inv'], np.nan)
        uc_old = uc_old.ix[self.holding_days, :]
        stacked_uc_old = uc_old.stack(dropna=True)

        f2 = plt.figure()
        ax2 = f2.add_subplot(1, 1, 1)
        plt.hist(stacked_uc_old.values)
        plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) + '/kde_old.png', dpi=1200)

        def pct_rank_qcut(x, *, n):
            if x.dropna().size <= 3:
                return pd.Series(np.nan, index=x.index)
            q_labels = pd.qcut(x, q=n, labels=False)
            return q_labels

        # 按照市值分组画图
        lncap = self.base.ix['lncap', self.holding_days, :]
        # 将市值分成3组
        lncap_labels = lncap.apply(pct_rank_qcut, axis=1, n=3)

        # 根据每组市值进行画图
        for i, mv in enumerate(['s', 'm', 'l']):
            # 根据市值标签所分的组
            curr_stacked_uc = unique_coverage.where(lncap_labels==i, np.nan).stack(dropna=True)
            curr_stacked_uc_old = uc_old.where(lncap_labels==i, np.nan).stack(dropna=True)

            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            plt.hist(curr_stacked_uc)
            plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) +
                        '/kde_' + mv + '.png', dpi=1200)

            f_old = plt.figure()
            ax_old = f_old.add_subplot(1, 1, 1)
            plt.hist(curr_stacked_uc_old)
            plt.savefig(str(os.path.abspath('.')) + '/' + str(self.strategy_data.stock_pool) +
                        '/kde_old_' + mv + '.png', dpi=1200)


    # 了解分析师报告中, 各项预测指标所占的比例
    def get_analyst_report_structure(self):
        self.db.initialize_jydb()
        self.db.initialize_sq()
        self.db.initialize_gg()
        self.db.get_trading_days()
        self.db.get_labels()

        sql_query = "select a.id, create_date, code, organ_id, author, Time_year, forecast_profit, " \
                    "forecast_income as revenue, forecast_income_share as eps, forecast_return, " \
                    "forecast_return_cash_share, forecast_return_capital_share from " \
                    "((select id, code, organ_id, author, create_date from DER_REPORT_RESEARCH where " \
                    "create_date>='" + str(self.db.trading_days.iloc[0]) + "' and create_date<='" + \
                    str(self.db.trading_days.iloc[-1]) + "') a " \
                    "left join (select * from DER_REPORT_SUBTABLE) b " \
                    "on a.id=b.report_search_id) " \
                    "order by create_date, code "
        original_data = self.db.gg_engine.get_original_data(sql_query)
        # pivot_data = original_data.pivot_table(index='create_date', columns='code', values=['forecast_profit',
        #                                         'revenue', 'eps', 'forecast_return', 'forecast_return_cash_share',
        #                                         'forecast_return_capital_share'])

        # 去除重复项
        unique_data = original_data.drop_duplicates(subset=['code', 'organ_id', 'author', 'Time_year'],
                                                    keep='last')
        # 各项指标的预测数量
        chara_fy_count = unique_data[['forecast_profit', 'revenue', 'eps', 'forecast_return',
                                'forecast_return_cash_share', 'forecast_return_capital_share']].count()
        report_fy_count = unique_data.shape[0]
        report_structure_unique = chara_fy_count / report_fy_count

        # 不去除重复项的版本
        report_structure = original_data.count()/original_data.shape[0]

        # 预测未来3个财年的报告, 占报告的比例
        # 这里的唯一报告是3个财年只算一个报告
        unique_report = original_data.drop_duplicates(subset=['code', 'organ_id', 'author'],
                                                      keep='last').shape[0]
        # 根据报告进行分组, 看每个报告中, 有预测3个财年的比例

        pass




    # 对原始数据的描述，即对论文的图表进行复制
    def data_description(self):
        self.get_table1a()
        self.get_table1b()
        self.get_fig2()
        self.get_table3()
        self.get_table6()
        # self.get_table7()
        self.draw_kde()
        pass

if __name__ == '__main__':
    ac = analyst_coverage()
    # ac.data_description()
    # ac.get_unique_coverage_number_parallel(rolling_days=120)
    # ac.get_unique_coverage_number()
    ac.get_analyst_report_structure()


































































