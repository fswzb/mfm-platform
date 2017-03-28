#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:07:18 2016

@author: lishiwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from datetime import datetime
import os

from data import data

# 数据类，所有数据均为pd.Panel, major_axis为时间，minor_axis为股票代码，items为数据名称

# 回测用到的数据类
class backtest_data(data):
    """ This is the data class used for back testing
    
    stock_price (pd.Panel): price data of stocks
    benchmark_price (pd.Panel): price data of benchmarks
    if_tradable (pd.Panel): information of whether a stock is enlisted/delisted or suspended from trading
    """
    
    def __init__(self):
        data.__init__(self)