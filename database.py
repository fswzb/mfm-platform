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
    def __init__(self, *, start_date = 'default', end_date = datetime.now().date().strftime('%Y-%m-%d'), market="83"):
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
        # 标记本次取数据是否为更新数据
        self.is_update = False

    # 初始化jydb
    def initialize_jydb(self):
        self.jydb_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='JYDB', add_info='')

    # 初始化sq
    def initialize_sq(self):
        self.sq_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                   server_ip='192.168.66.12', port='1433', db_name='SmartQuant', add_info='')

    # 初始化zyyx
    def initialize_gg(self):
        self.gg_engine = db_engine(server_type='mssql', driver='pymssql', username='lishi.wang', password='Zhengli1!',
                                     server_ip='192.168.66.12', port='1433', db_name='GOGOAL', add_info='')

    # 取交易日表
    def get_trading_days(self):
        sql_query = "select TradingDate as trading_days from QT_TradingDayNew where SecuMarket=" +\
                    self.market +" and IfTradingDay=1 "
        # 如果指定了开始结束日期，则选取开始结束日期之间的交易日
        if self.start_date != 'default':
            sql_query = sql_query + "and TradingDate>=" + "'" + str(self.start_date) + "' "
        if self.end_date != 'default':
            sql_query = sql_query + "and TradingDate<=" + "'" + str(self.end_date) + "' "
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
        self.data.const_data = pd.DataFrame(index=index_label)

    # 取ClosePrice_adj数据，将data中的panel数据index和columns都设置为ClosePrice_adj的index和columns
    # 先将所有的数据都取出来，之后不用再次从sq中取
    def get_ClosePrice_adj(self):
        sql_query = "select TradingDay, SecuCode, AdjustClosePrice as ClosePrice_adj, AdjustOpenPrice as OpenPrice_adj, " \
                    "OpenPrice, HighPrice, LowPrice, ClosePrice, RatioAdjustFactor as AdjustFactor, "\
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

    # 取open，close， high， low的价格数据
    def get_ochl(self):
        ochl = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice']
        for data_name in ochl:
            curr_data = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values=data_name)
            self.data.stock_price[data_name] = curr_data

    # 取Adjust Factor
    def get_AdjustFactor(self):
        AdjustFactor = self.sq_data.pivot_table(index='TradingDay', columns='SecuCode', values='AdjustFactor')
        self.data.stock_price['AdjustFactor'] = AdjustFactor

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
        self.data.raw_data['Industry'] = Industry

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
        list_status = list_status.pivot_table(index='ChangeDate',columns='SecuCode',values='ChangeType',
                                              aggfunc='first')
        # 更新数据的时候可能出现更新时间段没有新数据的情况，要处理这种情况
        if list_status.empty:
            list_status = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        # 向前填充
        list_status = list_status.fillna(method='ffill')

        # 上市标记为1，找到那些为1的，然后将false全改为nan，再向前填充true，即可得到is_enlisted
        # 即一旦上市后，之后的is_enlisted都为true
        is_enlisted = list_status == 1
        is_enlisted = is_enlisted.replace(False, np.nan)
        is_enlisted = is_enlisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_enlisted = is_enlisted.reindex(self.data.stock_price.major_axis, method='ffill')
        # 将股票索引对其，以保证fillna时可以填充所有的股票
        is_enlisted = is_enlisted.reindex(columns=self.data.stock_price.minor_axis)
        # 股票上市前会变成nan，它们未上市，因此将它们填成false
        # 更新的时候，那些一列全是nan的不能填，要等衔接旧数据时填
        if self.is_update:
            is_enlisted = is_enlisted.apply(lambda x:x if x.isnull().all() else x.fillna(0), axis=0)
        else:
            is_enlisted = is_enlisted.fillna(0).astype(np.int)

        # 退市标记为4， 找到那些为4的，然后将false改为nan，向前填充true，即可得到is_delisted
        # 即一旦退市之后，之后的is_delisted都为true
        # 退市准备期标记为6，都在4的前面，其他标记9，也在4的前面，而且两者数量很少，暂不考虑
        is_delisted = list_status == 4
        is_delisted = is_delisted.replace(False, np.nan)
        is_delisted = is_delisted.fillna(method='ffill')
        # 将时间索引和标准时间索引对齐，向前填充
        is_delisted = is_delisted.reindex(self.data.stock_price.major_axis, method='ffill')
        # 将股票索引对其，以保证fillna时可以填充所有的股票
        is_delisted = is_delisted.reindex(columns=self.data.stock_price.minor_axis)
        # 未退市过的股票，因为没有出现过4，会出现全是nan的情况，将它们填成false
        # 股票退市前会变成nan，它们未退市，依然填成false
        # 更新的时候，那些一列全是nan的不能填，要等衔接旧数据时填
        if self.is_update:
            is_delisted = is_delisted.apply(lambda x:x if x.isnull().all() else x.fillna(0), axis=0)
        else:
            is_delisted = is_delisted.fillna(0).astype(np.int)

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

        # 更新数据的时候可能出现更新时间段没有新数据的情况，要处理这种情况
        TotalAssets = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalAssets',
                                              aggfunc='first')
        if TotalAssets.empty:
            TotalAssets = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        else:
            TotalAssets = TotalAssets.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        TotalLiability = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalLiability',
                                                 aggfunc='first')
        if TotalLiability.empty:
            TotalLiability = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        else:
            TotalLiability = TotalLiability.fillna(method='ffill').reindex(self.data.stock_price.major_axis, method='ffill')
        TotalEquity = recent_data.pivot_table(index='InfoPublDate', columns='SecuCode', values='TotalEquity',
                                              aggfunc='first')
        if TotalEquity.empty:
            TotalEquity = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,
                                       columns=self.data.stock_price.minor_axis)
        else:
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
        sql_query = "set query_governor_cost_limit 0"\
                    "select b.DataDate, a.SecuCode, b.cash_earnings_ttm from " \
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
        sql_query = "set query_governor_cost_limit 0"\
                    "select b.DataDate, a.SecuCode, b.ni_ttm from " \
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
        sql_query = "set query_governor_cost_limit 0"\
                    "select b.DataDate, a.SecuCode, b.EndDate, b.ni_ttm, b.revenue_ttm, b.eps_ttm from " \
                    "(select distinct InnerCode, SecuCode from ReturnDaily) a " \
                    "left join (select DataDate, InnerCode, EndDate, NetProfit as ni_ttm, " \
                    "TotalOperatingRevenue as revenue_ttm, BasicEPS as eps_ttm from TTM_LC_IncomeStatementAll_8Q " \
                    "where DataDate>='" + str(self.trading_days.iloc[0]) + \
                    "' and DataDate<='" + str(self.trading_days.iloc[-1]) + "') b " \
                    "on a.InnerCode=b.InnerCode order by DataDate, SecuCode, EndDate"
        ttm_data_8q = self.sq_engine.get_original_data(sql_query)
        # 两年增长率，直接用每个时间点上的当前quarter的ttm数据除以8q以前的ttm数据减一
        grouped_data = ttm_data_8q.groupby(['DataDate', 'SecuCode'])
        growth_data = grouped_data['ni_ttm','revenue_ttm','eps_ttm'].apply(lambda x:(x.iloc[-1]/x.iloc[0])**(1/2)-1)
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
                    "where SecuCode in ('000001','000016','000300','000905','000906','H00016','H00300'," \
                    "'H00905','H00906') and SecuCategory=4) a "\
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
        for key in index_name:
            self.data.benchmark_price.ix['ClosePrice_'+index_name[key], :, 0] = index_close_price[key].values
            self.data.benchmark_price.ix['OpenPrice_'+index_name[key], :, 0] = index_open_price[key].values
        # 全收益指数只有收盘价, 而且没有上证综指的全收益
        index_adj_name = {'H00016':'adj_sz50', 'H00300':'adj_hs300', 'H00905':'adj_zz500', 'H00906':'adj_zz800'}
        for key in index_adj_name:
            self.data.benchmark_price.ix['ClosePrice_'+index_adj_name[key], :, 0] = index_close_price[key].values
        pass

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
        index_weight = weight_data.pivot_table(index='EndDate', columns=['index_code', 'comp_code'],
                                               values='Weight', aggfunc='first')
        index_name = {'000001': 'szzz', '000016': 'sz50', '000300': 'hs300', '000905': 'zz500', '000906': 'zz800'}
        # 更新数据的时候可能出现更新时间段没有新数据的情况，要处理这种情况
        if index_weight.empty:
            index_weight = pd.DataFrame(np.nan, index=self.data.stock_price.major_axis,columns=
                                        pd.MultiIndex.from_product([list(index_name.keys()), self.data.stock_price.minor_axis]))

        # 对指数进行循环储存
        for i in index_weight.columns.get_level_values(0).drop_duplicates():
            # 因为这里的数据，一行是所有指数数据都在一起，就会出现，如沪深300在这一期有数据，而中证500一个数据都没有的情况
            # 这种情况在其他数据中不会出现，其他数据是分开储存，如果一期都没有，那么原始数据就不会存在这一期，于是需要分开处理
            curr_weight = index_weight.ix[:, i]
            # 上面提到的更新数据时，出现的没有新数据的情况，不能dropna再填充，因为dropna会变成空dataframe
            if curr_weight.isnull().all().all():
                pass
            else:
                curr_weight = curr_weight.dropna(axis=0, how='all').reindex(curr_weight.index, method='ffill').\
                    reindex(self.data.stock_price.major_axis, method='ffill')
            self.data.benchmark_price['Weight_'+index_name[i]] = curr_weight
            # 将权重数据的nan填上0
            # 如果为更新数据，则一行全是nan的情况不填，一行有数据的情况才将nan填成0
            if self.is_update:
                self.data.benchmark_price['Weight_'+index_name[i]] = self.data.benchmark_price['Weight_'+index_name[i]].\
                    apply(lambda x:x if x.isnull().all() else x.fillna(0), axis=1)
            else:
                self.data.benchmark_price['Weight_'+index_name[i]] = \
                    self.data.benchmark_price['Weight_'+index_name[i]].fillna(0)
            pass

    # 从现在的因子库里取因子数据
    def get_existing_factor(self, factor_id):
        sql_query = "select runnerdate as TradingDay, stockticker as SecuCode, value as factor_value " \
                    "from RunnerValue where runnerdate>='" + str(self.trading_days.iloc[0]) + "' and " \
                    "runnerdate<='" + str(self.trading_days.iloc[-1]) + "' and runnerid=" + str(factor_id) + " " \
                    "order by TradingDay, SecuCode"
        existing_factor_data = self.sq_engine.get_original_data(sql_query)
        existing_factor = existing_factor_data.pivot_table(index='TradingDay', columns='SecuCode', values='factor_value')

        # 储存数据
        self.data.stock_price['existing_factor'] = existing_factor

    # 储存数据文件
    def save_data(self):
        data.write_data(self.data.stock_price)
        data.write_data(self.data.raw_data)
        data.write_data(self.data.benchmark_price)
        data.write_data(self.data.if_tradable)
        self.data.const_data.to_csv('const_data.csv', index_label='datetime', na_rep='NaN', encoding='GB18030')

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
        self.get_ochl()
        self.get_AdjustFactor()
        self.get_Volume()
        self.get_total_and_free_mv()
        self.get_total_and_free_shares()
        self.get_Industry()
        self.get_is_suspended()
        print('get sq data has been completed...\n')
        self.get_list_status(first_date=update_time)
        print('get list status has been completed...\n')
        self.get_asset_liability_equity(first_date=update_time)
        print('get balancesheet data has been completed...\n')
        self.get_pb()
        self.get_ni_fy1_fy2()
        self.get_eps_fy1_fy2()
        print('get forecast data has been completed...\n')
        self.get_cash_earings_ttm()
        print('get cash earnings ttm has been completed...\n')
        self.get_ni_ttm()
        print('get netincome ttm has been completed...\n')
        self.get_pe_ttm()
        self.get_ni_revenue_eps_growth()
        print('get growth ttm has been completed...\n')
        self.get_index_price()
        self.get_index_weight(first_date=update_time)
        print('get index data has been completed...\n')

        # 更新数据的情况先不能储存数据，只有非更新的情况才能储存
        if not self.is_update:
           self.save_data()

    # 更新数据的主函数
    def update_data_from_db(self, *, end_date='default'):
        # 更新标记
        self.is_update = True
        # 首先读取ClosePrice_adj数据，将其当作更新数据时的参照标签
        data_mark = pd.read_csv('ClosePrice_adj.csv', parse_dates=True, index_col=0)
        # 更新的第一天为之前数据标签日期的最后一天
        # 因为有可能当时更新数据的时候，还没有得到那次的数据
        # 因此为了统一，更新的第一天都设置为那一天
        last_day = data_mark.iloc[-1].name
        self.start_date = last_day

        # 可以设置更新数据的更新截止日，默认为更新到当天
        if end_date != 'default':
            self.end_date = end_date

        # 更新数据
        self.get_data_from_db(update_time=self.start_date)

        # 读取以前的老数据
        stock_price_name_list = ['ClosePrice_adj', 'OpenPrice_adj', 'OpenPrice', 'ClosePrice', 'HighPrice',
                                 'LowPrice', 'AdjustFactor', 'Volume', 'Shares', 'FreeShares',
                                 'MarketValue', 'FreeMarketValue']
        raw_data_name_list = ['Industry', 'TotalAssets', 'TotalLiability', 'TotalEquity', 'PB', 'NetIncome_fy1',
                              'NetIncome_fy2', 'EPS_fy1', 'EPS_fy2', 'CashEarnings_ttm', 'NetIncome_ttm',
                              'PE_ttm', 'NetIncome_ttm_growth_8q', 'Revenue_ttm_growth_8q', 'EPS_ttm_growth_8q']
        if_tradable_name_list = ['is_suspended', 'is_enlisted', 'is_delisted']
        benchmark_index_name = ['szzz', 'sz50', 'hs300', 'zz500', 'zz800']
        benchmark_data_type = ['ClosePrice', 'OpenPrice', 'Weight', 'ClosePrice_adj']
        benchmark_price_name_list = [a+'_'+b for a in benchmark_data_type for b in benchmark_index_name]
        # 注意上证综指没有closeprice adj, 即全收益数据
        benchmark_price_name_list.remove('ClosePrice_adj_szzz')

        old_stock_price = data.read_data(stock_price_name_list, stock_price_name_list)
        old_raw_data = data.read_data(raw_data_name_list, raw_data_name_list)
        old_if_tradable = data.read_data(if_tradable_name_list, if_tradable_name_list)
        old_benchmark_price = data.read_data(benchmark_price_name_list, benchmark_price_name_list)
        old_const_data = data.read_data(['const_data'], ['const_data'])
        old_const_data = old_const_data.ix['const_data']

        # 新数据中的股票数可能与旧数据已经不同，要将旧数据中的股票索引换成新数据的索引
        new_stock_index = self.data.stock_price.minor_axis
        old_stock_price = old_stock_price.reindex(minor_axis=new_stock_index)
        old_raw_data = old_raw_data.reindex(minor_axis=new_stock_index)
        old_if_tradable = old_if_tradable.reindex(minor_axis=new_stock_index)
        old_benchmark_price = old_benchmark_price.reindex(minor_axis=new_stock_index)

        # 衔接新旧数据
        new_stock_price = pd.concat([old_stock_price.drop(last_day, axis=1).sort_index(),
                                     self.data.stock_price.sort_index()], axis=1)
        new_raw_data = pd.concat([old_raw_data.drop(last_day, axis=1).sort_index(),
                                     self.data.raw_data.sort_index()], axis=1)
        new_if_tradable = pd.concat([old_if_tradable.drop(last_day, axis=1).sort_index(),
                                     self.data.if_tradable.sort_index()], axis=1)
        new_benchmark_price = pd.concat([old_benchmark_price.drop(last_day, axis=1).sort_index(),
                                     self.data.benchmark_price.sort_index()], axis=1)
        new_const_data = pd.concat([old_const_data.drop(last_day, axis=0).sort_index(axis=1),
                                    self.data.const_data.sort_index(axis=1)], axis=0)

        self.data.stock_price = new_stock_price
        self.data.raw_data = new_raw_data
        self.data.if_tradable = new_if_tradable
        self.data.benchmark_price = new_benchmark_price
        self.data.const_data = new_const_data

        # 对需要的数据进行fillna，以及fillna后的重新计算
        # 主要是那些用到first_date参数的数据，以及涉及这些数据的衍生数据
        self.data.raw_data['TotalAssets'] = self.data.raw_data['TotalAssets'].fillna(method='ffill')
        self.data.raw_data['TotalLiability'] = self.data.raw_data['TotalLiability'].fillna(method='ffill')
        self.data.raw_data['TotalEquity'] = self.data.raw_data['TotalEquity'].fillna(method='ffill')
        self.get_pb()
        # 注意这两个数据在用旧数据向前填na之后，还要再fill一次na，因为更新的时候出现的新股票，之前的旧数据因为重索引的关系，也是nan
        self.data.if_tradable['is_enlisted'] = self.data.if_tradable['is_enlisted'].\
            fillna(method='ffill').fillna(0).astype(np.int)
        self.data.if_tradable['is_delisted'] = self.data.if_tradable['is_delisted'].\
            fillna(method='ffill').fillna(0).astype(np.int)
        for index_name in benchmark_index_name:
            self.data.benchmark_price['Weight_'+index_name] = self.data.benchmark_price['Weight_'+index_name]\
                                                              .fillna(method='ffill')
        self.save_data()

        # 重置标记
        self.is_update = False

if __name__ == '__main__':
    import time
    start_time = time.time()
    db = database(start_date='2007-01-01', end_date='2017-04-27')
    # db.get_data_from_db()
    db.update_data_from_db(end_date='2017-06-14')
    # db.initialize_jydb()
    # db.initialize_sq()
    # db.initialize_gg()
    # db.get_trading_days()
    # db.get_labels()
    # db.get_ClosePrice_adj()
    # db.get_index_price()
    # data.write_data(db.data.benchmark_price)
    print("time: {0} seconds\n".format(time.time()-start_time))













































































