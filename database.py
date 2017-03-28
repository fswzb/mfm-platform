#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:13:36 2017

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data
from db_engine import db_engine

# 维护数据库的类

class database(object):
    """ This is the class of handling database, including fetch and update data from database

    foo
    """
    def __init__(self, *, start_date = 'default', end_date = 'default', market="83"):
        # 储存交易日表
        self.trading_days = pd.Series()
        # 数据库取出来后整理成型的数据
        self.data = data()
        # 聚源数据库的引擎
        self.jydb_engine = 'NOT initialized yet!'
        # smart quant数据库引擎，取常用的行情数据
        self.sq_engine = 'NOT initialized yet!'
        # smart quant数据库中取出的数据
        self.sq_data = pd.DataFrame()
        # 朝阳永续数据库引擎，取分析师预期数据
        self.gg_engine = 'NOT initialized yet!'
        # 所取数据的开始、截止日期，市场代码
        self.start_date = start_date
        self.end_date = end_date
        self.market = market

    # 初始化jydb
    def initialize_jydb(self):
        self.jydb_engine = db_engine(server_type='mssql', driver='pyodbc', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='JYDB', add_info='driver=FreeTDS')

    # 初始化sq
    def initialize_sq(self):
        self.sq_engine = db_engine(server_type='mssql', driver='pyodbc', username='lishi.wang', password='Zhengli1!',
                                   server_ip='192.168.66.12', port='1433', db_name='SmartQuant', add_info='driver=FreeTDS')

    # 初始化zyyx
    def initialize_gg(self):
        self.gg_engine = db_engine(server_type='mssql', driver='pyodbc', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='GOGOAL', add_info='driver=FreeTDS')

    # 取交易日表
    def get_trading_days(self):
        sql_query = "select TradingDate as trading_days from QT_TradingDayNew where SecuMarket=" +\
                    self.market +" and IfTradingDay=1 "
        # 如果指定了开始结束日期，则选取开始结束日期之间的交易日
        if self.start_date != 'default':
            sql_query = sql_query + "and TradingDate>=" + "'" + self.start_date + "' "
        if self.end_date != 'default':
            sql_query = sql_query + "and TradingDate<=" + "'" + self.end_date + "' "
        sql_query = sql_query + 'order by trading_days'

        # 取数据
        trading_days = self.jydb_engine.get_original_data(sql_query)
        self.trading_days = trading_days['trading_days']

    # 设定数据的index和columns，index以交易日表为准，columns以sq中的return daily里的股票为准
    def get_labels(self):
        sql_query = 'select distinct SecuCode from ReturnDaily order by SecuCode'
        column_label = self.sq_engine.get_original_data(sql_query)
        column_label = column_label.ix[:, 0]
        index_label = self.trading_days

        # data中的所有交易日和股票数据都以这两个label为准，包括benchmark
        self.data.stock_price = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.raw_data = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.benchmark_price = pd.Panel(major_axis=index_label, minor_axis=column_label)
        self.data.if_tradable = pd.Panel(major_axis=index_label, minor_axis=column_label)

    # 取ClosePrice_adj数据，将data中的panel数据index和columns都设置为ClosePrice_adj的index和columns
    # 先将所有的数据都取出来，之后不用再次从sq中取
    def get_ClosePrice_adj(self):
        sql_query = "select TradingDay, SecuCode, AdjustClosePrice as ClosePrice_adj, AdjustOpenPrice as OpenPrice_adj, "\
                    "TurnoverVolume as Volume, TotalShares as Shares, NonRestrictedShares as FreeShares, "\
                    "MarketCap as MarketValue, FloatMarketCap as FreeMarketValue, IndustryNameNew as Industry, "\
                    "IfSuspended as is_suspended "\
                    "from ReturnDaily where "\
                    "IfTradingDay=1 and TradingDay>='" + str(self.trading_days.iloc[0]) + "' and TradingDay<='" + \
                    str(self.trading_days.iloc[-1]) + "' order by TradingDay, SecuCode"
        self.sq_data = self.sq_engine.get_original_data(sql_query)
        ClosePrice_adj = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='ClosePrice_adj')

        # 储存ClosePrice_adj
        self.data.stock_price['ClosePrice_adj'] = ClosePrice_adj

    # 取OpenPrice_adj
    def get_OpenPrice_adj(self):
        OpenPrice_adj = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='OpenPrice_adj')
        self.data.stock_price['OpenPrice_adj'] = OpenPrice_adj

    # 取volumne
    def get_Volume(self):
        Volume = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='Volume')
        self.data.stock_price['Volume'] = Volume

    # 取total shares和free shares
    def get_total_and_free_shares(self):
        Shares = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='Shares')
        self.data.stock_price['Shares'] = Shares
        FreeShares = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='FreeShares')
        self.data.stock_price['FreeShares'] = FreeShares

    # 取total mv和free mv
    def get_total_and_free_mv(self):
        MarketValue = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='MarketValue')
        self.data.stock_price['MarketValue'] = MarketValue
        FreeMarketValue = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='FreeMarketValue')
        self.data.stock_price['FreeMarketValue'] = FreeMarketValue

    # 取行业标签
    def get_Industry(self):
        Industry = self.sq_data.pivot(index='TradingDay', columns='SecuCode', values='Industry')
        self.data.stock_price['Industry'] = Industry

    # 取是否停牌
    def get_is_suspended(self):
        is_suspended = self.sq_data.pivot(index='TradingDay', columns='SecuCode', values='is_suspended')
        self.data.if_tradable['is_suspended'] = is_suspended

    # 取上市退市标记，即if_enlisted & if_delisted
    # 如果是第一次取数据（非更新数据）一些数据（包括财务数据）的起始日期并不是第一个交易日，
    # 即第一个交易日的数据在数据库里并不是标记为这个交易日的数据
    # 而是之前的数据，因此在非更新数据的情况下，起始日期选取为一个最小日期，以保证取到所有数据
    def get_list_status(self, *, first_date=pd.Timestamp('1900-01-01')):
        sql_query = "select a.SecuCode, b.ChangeDate, b.ChangeType from "\
                    "(select distinct InnerCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select ChangeDate, ChangeType, InnerCode from LC_ListStatus where SecuMarket in " \
                    "(83,90) and ChangeDate>='" + str(first_date) + "' and ChangeDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b on a.InnerCode=b.InnerCode "\
                    " order by SecuCode, ChangeDate"
        list_status = self.jydb_engine.get_original_data(sql_query)
        list_status = list_status.pivot_table(index='ChangeDate',columns='SecuCode',values='ChangeType')
        # 向前填充
        list_status = list_status.fillna(method='ffill')

        # 上市标记为1，找到那些为1的，然后将false全改为nan，再向前填充true，即可得到is_enlisted
        # 即一旦上市后，之后的is_enlisted都为true
        is_enlisted = list_status == 1
        is_enlisted = is_enlisted.replace(False, np.nan)
        is_enlisted = is_enlisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_enlisted = is_enlisted.reindex(self.data.stock_price.major_axis, method='ffill')
        # 股票上市前会变成nan，它们未上市，因此将它们填成false
        is_enlisted = is_enlisted.fillna(0)

        # 退市标记为4， 找到那些为4的，然后将false改为nan，向前填充true，即可得到is_delisted
        # 即一旦退市之后，之后的is_delisted都为true
        # 退市准备期标记为6，都在4的前面，其他标记9，也在4的前面，而且两者数量很少，暂不考虑
        is_delisted = list_status == 4
        is_delisted = is_delisted.replace(False, np.nan)
        is_delisted = is_delisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_delisted = is_delisted.reindex(self.data.stock_price.major_axis, method='ffill')
        # 未退市过的股票，因为没有出现过4，会出现全是nan的情况，将它们填成false
        # 股票退市前会变成nan，它们未退市，依然填成false
        is_delisted = is_delisted.fillna(0)

        self.data.if_tradable['is_enlisted'] = is_enlisted
        self.data.if_tradable['is_delisted'] = is_delisted

    # 取总资产，总负债和所有者权益
    # 取合并报表，即if_merged = 1
    # 报表会进行调整因此每个时间点上可能会有多个不同时间段的报表，类似于前复权
    def get_asset_liability_equity(self, *, first_date=pd.Timestamp('1900-01-01')):
        sql_query = "select b.InfoPublDate, b.EndDate, a.SecuCode, b.TotalAssets, b.TotalLiability, "\
                    "b.TotalEquity from ("\
                    "select distinct CompanyCode, SecuCode from SmartQuant.dbo.ReturnDaily) a " \
                    "left join (select InfoPublDate, EndDate, CompanyCode, TotalAssets, TotalLiability, " \
                    "TotalShareholderEquity as TotalEquity from LC_BalanceSheetAll where IfMerged=1 "\
                    "and InfoPublDate>='" + str(first_date) + "' and InfoPublDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b on a.CompanyCode=b.CompanyCode "\
                    " order by InfoPublDate, SecuCode, EndDate"
        balance_sheet_data = self.jydb_engine.get_original_data(sql_query)

        # 对资产负债和所有者权益，只取每个时间点上最近的那一期报告，
        # 因为每个时间点上只会使用当前时间点的最新值，不是涉及变化率的计算
        recent_data = balance_sheet_data.groupby(['InfoPublDate', 'SecuCode'],as_index=False).nth(-1)

        TotalAssets = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalAssets')
        TotalAssets = TotalAssets.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        TotalLiability = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalLiability')
        TotalLiability = TotalLiability.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        TotalEquity = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalEquity')
        TotalEquity = TotalEquity.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')

        self.data.raw_data['TotalAssets'] = TotalAssets
        self.data.raw_data['TotalLiability'] = TotalLiability
        self.data.raw_data['TotalEquity'] = TotalEquity

    # 计算pb
    def get_pb(self):
        pb = self.data.stock_price.ix['FreeMarketValue']/self.data.raw_data.ix['TotalEquity']
        self.data.raw_data['PB'] = pb

    # 取一致预期净利润
    def get_ni_fy1_fy2(self):
        sql_query = "select STOCK_CODE, CON_DATE, C4*10000 as NI, "\
                    "ROW_NUMBER() over (partition by stock_code, con_date order by rpt_date) as fy from "\
                    "CON_FORECAST_STK where C4_TYPE!=0 and con_date>='" + str(self.trading_days.iloc[0]) + \
                    "' and con_date<='" + str(self.trading_days.iloc[-1]) + \
                    "' order by stock_code, con_date, rpt_date"
        forecast_ni = self.gg_engine.get_original_data(sql_query)
        grouped_data = forecast_ni.groupby(['CON_DATE', 'STOCK_CODE'], as_index=False)
        fy1_data = grouped_data.nth(0)
        fy2_data = grouped_data.nth(1)
        ni_fy1 = fy1_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='NI')
        ni_fy2 = fy2_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='NI')
        ni_fy1 = ni_fy1.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        ni_fy2 = ni_fy2.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')

        self.data.raw_data['NetIncome_fy1'] = ni_fy1
        self.data.raw_data['NetIncome_fy2'] = ni_fy2

    # 取一致预期eps
    def get_eps_fy1_fy2(self):
        sql_query = "select STOCK_CODE, CON_DATE, C1 as EPS, " \
                    "ROW_NUMBER() over (partition by stock_code, con_date order by rpt_date) as fy from " \
                    "CON_FORECAST_STK where CON_TYPE!=0 and con_date>='" + str(self.trading_days.iloc[0]) + \
                    "' and con_date<='" + str(self.trading_days.iloc[-1]) + \
                    "' order by stock_code, con_date, rpt_date"
        forecast_eps = self.gg_engine.get_original_data(sql_query)
        grouped_data = forecast_eps.groupby(['CON_DATE', 'STOCK_CODE'], as_index=False)
        fy1_data = grouped_data.nth(0)
        fy2_data = grouped_data.nth(1)
        eps_fy1 = fy1_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='EPS')
        eps_fy2 = fy2_data.pivot_table(index='CON_DATE', columns='STOCK_CODE', values='EPS')
        eps_fy1 = eps_fy1.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        eps_fy2 = eps_fy2.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')

        self.data.raw_data['EPS_fy1'] = eps_fy1
        self.data.raw_data['EPS_fy2'] = eps_fy2

    # 取cash earnings ttm
    def get_cash_earings_ttm(self):
        sql_query = "select b.DataDate, a.SecuCode, b.cash_earnings_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a left join " \
                    "(select DataDate, CashEquivalentIncrease as cash_earnings_ttm, InnerCode from " \
                    "TTM_LC_CashFlowStatementAll where DataDate>='" + str(self.trading_days.iloc[0]) + \
                    "' and DataDate<='" + str(self.trading_days.iloc[-1]) + "') b " \
                    "on a.InnerCode=b.InnerCode order by DataDate, SecuCode"
        ttm_data = self.sq_engine.get_original_data(sql_query)
        cash_earnings_ttm = ttm_data.pivot_table(index='DataDate', columns='SecuCode', values='cash_earnings_ttm')
        cash_earnings_ttm = cash_earnings_ttm.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                             method='ffill')
        self.data.raw_data['CashEarnings_ttm'] = cash_earnings_ttm

    # 取net income ttm
    def get_ni_ttm(self):
        sql_query = "select b.DataDate, a.SecuCode, b.ni_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a left join " \
                    "(select DataDate, NetProfit as ni_ttm, InnerCode from TTM_LC_IncomeStatementAll " \
                    "where DataDate>='" + str(self.trading_days.iloc[0]) + "' and DataDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b on a.InnerCode=b.InnerCode " \
                    "order by DataDate, SecuCode"
        ttm_data = self.sq_engine.get_original_data(sql_query)
        ni_ttm = ttm_data.pivot_table(index='DataDate', columns='SecuCode', values='ni_ttm')
        ni_ttm = ni_ttm.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                             method='ffill')
        self.data.raw_data['NetIncome_ttm'] = ni_ttm

    # 计算pe ttm
    def get_pe_ttm(self):
        pe_ttm = self.data.stock_price.ix['FreeMarketValue']/self.data.raw_data.ix['NetIncome_ttm']
        self.data.raw_data['PE_ttm'] = pe_ttm

    # 取ni ttm, revenue ttm, eps_ttm的两年增长率
    def get_ni_revenue_eps_growth(self):
        sql_query = "select b.DataDate, a.SecuCode, b.EndDate, b.ni_ttm, b.revenue_ttm, b.eps_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a " \
                    "left join (select DataDate, InnerCode, EndDate, NetProfit as ni_ttm, " \
                    "TotalOperatingRevenue as revenue_ttm, BasicEPS as eps_ttm from TTM_LC_IncomeStatementAll_8Q " \
                    "where DataDate>='" + str(self.trading_days.iloc[0]) + \
                    "' and DataDate<='" + str(self.trading_days.iloc[-1]) + "') b " \
                    "on a.InnerCode=b.InnerCode order by DataDate, SecuCode, EndDate"
        ttm_data_8q = self.sq_engine.get_original_data(sql_query)
        # 两年增长率，直接用每个时间点上的当前quarter的ttm数据除以8q以前的ttm数据减一
        grouped_data = ttm_data_8q.groupby(['DataDate', 'SecuCode'])
        growth_data = grouped_data['ni_ttm','revenue_ttm','eps_ttm'].apply(lambda x:x.iloc[-1]/x.iloc[0]-1)
        time_index = growth_data.index.get_level_values(0)
        stock_index = growth_data.index.get_level_values(1)

        ni_ttm_growth_8q = growth_data.pivot_table(index=time_index, columns=stock_index, values='ni_ttm')
        ni_ttm_growth_8q = ni_ttm_growth_8q.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                           method='ffill').replace(np.inf, np.nan)
        revenue_ttm_growth_8q = growth_data.pivot_table(index=time_index, columns=stock_index, values='revenue_ttm')
        revenue_ttm_growth_8q = revenue_ttm_growth_8q.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                           method='ffill').replace(np.inf, np.nan)
        eps_ttm_growth_8q = growth_data.pivot_table(index=time_index, columns=stock_index, values='eps_ttm')
        eps_ttm_growth_8q = eps_ttm_growth_8q.fillna(method='ffill').reindex(self.data.stock_price.major_axis,
                                                                           method='ffill').replace(np.inf, np.nan)
        self.data.raw_data['NetIncome_ttm_growth_8q'] = ni_ttm_growth_8q
        self.data.raw_data['Revenue_ttm_growth_8q'] = revenue_ttm_growth_8q
        self.data.raw_data['EPS_ttm_growth_8q'] = eps_ttm_growth_8q

    # 取指数行情数据
    def get_index_price(self):
        sql_query = "select b.TradingDay, a.SecuCode, b.ClosePrice, b.OpenPrice from "\
                    "(select distinct InnerCode, SecuCode from SecuMain "\
                    "where SecuCode in ('000001','000016','000300','000905','000906') and SecuCategory=4) a "\
                    "left join (select InnerCode, TradingDay, ClosePrice, OpenPrice from QT_IndexQuote "\
                    "where TradingDay>='" + str(self.trading_days.iloc[0]) + "' and TradingDay<='" + \
                    str(self.trading_days.iloc[-1]) + "') b "\
                    "on a.InnerCode=b.InnerCode order by TradingDay, SecuCode"
        index_data = self.jydb_engine.get_original_data(sql_query)
        index_close_price = index_data.pivot_table(index='TradingDay', columns='SecuCode', values='ClosePrice')
        index_close_price = index_close_price.reindex(self.data.stock_price.major_axis)
        index_open_price = index_data.pivot_table(index='TradingDay', columns='SecuCode', values='OpenPrice')
        index_open_price = index_open_price.reindex(self.data.stock_price.major_axis)
        # 鉴于指数行情的特殊性，将指数行情都存在benchmark price中的每个item的第一列
        index_name = {'000001':'szzz', '000016':'sz50', '000300':'hs300', '000905':'zz500', '000906':'zz800'}
        for column in index_close_price:
            self.data.benchmark_price.ix['ClosePrice_'+index_name[column], :, 0] = index_close_price[column].values
            self.data.benchmark_price.ix['OpenPrice_'+index_name[column], :, 0] = index_open_price[column].values

    # 取指数权重数据
    def get_index_weight(self, *, first_date=pd.Timestamp('1900-01-01')):
        sql_query = "select b.EndDate, a.SecuCode as index_code, c.SecuCode as comp_code, b.Weight/100 as Weight from "\
                    "(select distinct InnerCode, SecuCode from SecuMain "\
                    "where SecuCode in ('000001','000016','000300','000905','000906') and SecuCategory=4) a "\
                    "left join (select EndDate, IndexCode, InnerCode, Weight from LC_IndexComponentsWeight "\
                    "where EndDate>='" + str(first_date) + "' and EndDate<='" + \
                    str(self.trading_days.iloc[-1]) + "') b "\
                    "on a.InnerCode=b.IndexCode "\
                    "left join (select distinct InnerCode, SecuCode from SecuMain) c "\
                    "on b.InnerCode=c.InnerCode "\
                    "order by EndDate, index_code, comp_code "
        weight_data = self.jydb_engine.get_original_data(sql_query)
        index_weight = weight_data.pivot_table(index='EndDate', columns=['index_code', 'comp_code'], values='Weight')
        index_weight = index_weight.reindex(self.data.stock_price.major_axis,
                                                                   method='ffill')
        index_name = {'000001': 'szzz', '000016': 'sz50', '000300': 'hs300', '000905': 'zz500', '000906': 'zz800'}
        # 对指数进行循环储存
        for i in index_weight.columns.get_level_values(0).drop_duplicates():
            self.data.benchmark_price['Weight_'+index_name[i]] = index_weight.ix[:, i]
            # 将权重数据的nan填上0
            self.data.benchmark_price['Weight_'+index_name[i]] = self.data.benchmark_price['Weight_'+index_name[i]].fillna(0)

    # 储存数据文件
    def save_data(self):
        data.write_data(self.data.stock_price)
        data.write_data(self.data.raw_data)
        data.write_data(self.data.benchmark_price)
        data.write_data(self.data.if_tradable)

    # 取数据的主函数
    # update_time为default时，则为首次取数据，需要更新数据时，传入更新的第一个交易日的时间给update_time即可
    def get_data_from_db(self, *, update_time=pd.Timestamp('1900-01-01')):
        self.initialize_jydb()
        self.initialize_sq()
        self.initialize_gg()
        self.get_trading_days()
        self.get_labels()
        self.get_ClosePrice_adj()
        self.get_OpenPrice_adj()
        self.get_Volume()
        self.get_total_and_free_mv()
        self.get_total_and_free_shares()
        self.get_Industry()
        self.get_is_suspended()
        self.get_list_status(first_date=update_time)
        self.get_asset_liability_equity(first_date=update_time)
        self.get_pb()
        self.get_ni_fy1_fy2()
        self.get_eps_fy1_fy2()
        self.get_cash_earings_ttm()
        self.get_ni_ttm()
        self.get_pe_ttm()
        self.get_ni_revenue_eps_growth()
        self.get_index_price()
        self.get_index_weight(first_date=update_time)

        self.save_data()


if __name__ == '__main__':
    import time
    from multiprocessing import Process
    start_time = time.time()
    db = database(start_date='2017-01-01',end_date=datetime.now().date().strftime('%Y-%m-%d'))
    
    print("time: {0} seconds\n".format(time.time()-start_time))













































































