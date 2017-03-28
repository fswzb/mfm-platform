#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:06:26 2017

@author: lishiwang
"""

from sqlalchemy import create_engine
import pandas as pd

# 建立数据库引擎的类

class db_engine(object):
    """This is the class of database engine.
    
    foo
    """
    def __init__(self, *, server_type, driver, username, password, server_ip, port, db_name, 
                 add_info = ''):
        # 创建引擎的string
        self.engine_str = server_type+'+'+driver+'://'+username+':'+password+'@'+server_ip+':'+ \
                          port+'/'+db_name 
        if add_info != '':
            self.engine_str = self.engine_str+'?'+add_info
        # 创建引擎
        self.engine = create_engine(self.engine_str)
        
    # 取数据
    def get_original_data(self, sql_query):
        """ Fetch data from this database engine using specified sql query.

        :param sql_query: (string) sql query which used to fetch data from this database engine
        :return: (pd.DataFrame) outcome data with time as index and fields as columns
        """
        data = pd.read_sql(sql_query, self.engine)
        return data
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        