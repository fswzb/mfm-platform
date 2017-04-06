#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:17:06 2017

@author: lishiwang
"""

from single_factor_strategy import single_factor_strategy
import pandas as pd

# 建立单因子策略对象
sf = single_factor_strategy()

#sf.single_factor_test(factor='FreeMarketValue', direction='-', bkt_start=pd.Timestamp('2009-03-03'),
#                      bkt_end=pd.Timestamp('2017-03-30'))

sf.sf_test_multiple_pools(factor='FreeMarketValue', direction='-', bkt_start=pd.Timestamp('2009-03-03'),
                          bkt_end=pd.Timestamp('2017-03-30'))