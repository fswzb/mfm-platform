#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:29:49 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

# 储存持仓的持仓类

class position(object):
    """ This is the class of holding matrix.
    
    holding_matrix (pd.DataFrame) : the holding_matrix of this position
    """
    
    def __init__(self, standard = 'default'):
        if standard is 'default':
            self.holding_matrix = pd.DataFrame()
        else:
            self.holding_matrix = pd.DataFrame(np.zeros(standard.shape), 
                                               index = standard.index, 
                                               columns = standard.columns)
        
    # 根据某一指标，对持仓进行加权，如对市值进行加权
    def weighted_holding(self, weights):
        """ Get the weighted holding matrix
        
        foo
        """
        self.holding_matrix = self.holding_matrix.mul(weights, fill_value = 0)
        self.to_percentage()
        pass

    # 根据行业标签，进行分行业加权，可以选择行业内如何加权，以及行业间如何加权
    def weighted_holding_indus(self, industry, *, inner_weights=0, outter_weights=0):
        # 定义行业内加权的函数
        def inner_weighting(grouped_data):
            new_holding = grouped_data['holding'].mul(grouped_data['inner_weights'], fill_value=0)
            if not (new_holding==0).all():
                new_holding = new_holding.div(new_holding.sum())
            return new_holding
        # 如果行业内权重为0，则为行业内等权
        if type(inner_weights) == int and inner_weights == 0:
            inner_weights = pd.DataFrame(1, index=self.holding_matrix.index, columns=self.holding_matrix.columns)
        # 根据持仓日循环
        for time, curr_holding in self.holding_matrix.iterrows():
            curr_data = pd.DataFrame({'holding':curr_holding,
                                      'inner_weights':inner_weights.ix[time]})
            # 处理行业数据最后一天可能全是nan的特殊情况（易在取数据时出现）
            if industry.ix[time].isnull().all():
                continue
            grouped = curr_data.groupby(industry.ix[time])
            # 进行行业内加权
            after_inner = grouped.apply(inner_weighting)
            # 对行业间加权数据进行求和
            # 如果行业间权重为0，则为行业间等权，即每个行业的总权重为1，（注意不是每个股票的权重为1）
            if type(outter_weights) == int and outter_weights == 0:
                outter_weights_sum = pd.Series(1, index=after_inner.index)
            else:
                outter_weights_sum = outter_weights.ix[time].groupby(industry.ix[time]).transform('sum')
            after_outter = after_inner.mul(outter_weights_sum).fillna(0)
            print(time)
            self.holding_matrix.ix[time] = after_outter.reset_index(level=0, drop=True)
        self.holding_matrix = self.holding_matrix.fillna(0)
        self.to_percentage()
        pass


    # 将持仓归一化，成为加总为1的百分比数
    def to_percentage(self):
        # 注意如果一期持仓全是0，则不改动
        self.holding_matrix = self.holding_matrix.apply(lambda x:x if (x==0).all() else
                                                        x.div(x.sum()), axis=1)
        # 防止无持仓的变成nan
        self.holding_matrix[self.holding_matrix.isnull()] = 0
        
    # 添加持股的函数，即，将选出的股票加入到对应时间点的持仓中去
    def add_holding(self, time, to_be_added):
        """ Add the holding matrix with newly selected(or bought) stocks.
        
        foo
        """
        self.holding_matrix.ix[time] = self.holding_matrix.ix[time].add(to_be_added, fill_value = 0)
     
    # 减少持股的函数，即，将选出减持的股票从对应的时间点持仓中减去
    def subtract_holding(self, time, to_be_subtracted):
        """ Subtract the holding matrix with newly subtracted(or sold) stocks.
        
        foo
        """
        self.holding_matrix.ix[time] = self.holding_matrix.ix[time].sub(to_be_subtracted, fill_value = 0)


































             
        
        