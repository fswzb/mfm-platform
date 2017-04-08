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
        
    # 将持仓归一化，成为加总为1的百分比数
    def to_percentage(self):
        # 注意如果一期持仓全是0，则不改动
        self.holding_matrix = self.holding_matrix.apply(lambda x:x if (x==0).all() else
                                                        x.div(x.sum()), axis=1)
        self.holding_matrix = self.holding_matrix.div(self.holding_matrix.sum(1), axis = 0)
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


































             
        
        