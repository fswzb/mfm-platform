import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('PDF')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import copy
from matplotlib.backends.backend_pdf import PdfPages
from cvxopt import solvers, matrix
from sklearn.decomposition import PCA
from scipy import optimize

from data import data
from strategy_data import strategy_data
from position import position
from strategy import strategy
from backtest import backtest
from barra_base import barra_base
from performance import performance
from performance_attribution import performance_attribution

# 处理净值表的类
class fof_handler(object):
    def __init__(self):
        self.nav = pd.Series()
        self.stock_value = pd.Series()
        self.stock_value_change = pd.Series()
        self.stock_position = position()
        self.full_dir = ''

    # 从目录中读取xlsx文件
    def read_xlsx(self, *, dir_name):
        pass

    # 处理excel文件
    def handle_xls(self, f):
        print('Please customize your own function of handling xls files in the derived class!\n')
        pass

    # 根据净值序列计算
    def handle_nav(self):
        # 建立画pdf的对象
        self.pdfs = PdfPages(self.full_dir[:-1] + '/allfigs.pdf')
        self.performance = performance(self.nav)
        self.performance.get_performance(foldername=self.folder_name[:-1])
        self.performance.plot_performance(foldername=self.folder_name[:-1], pdfs=self.pdfs)

    # 根据股票持仓进行回测归因
    def handle_stock_position(self):
        # 将stock position的股票代码改成标准化的形式

        # 将股票持仓的日期改为nav的对应日期, 以防其这段时间没有股票持仓
        self.stock_position.holding_matrix = self.stock_position.holding_matrix.\
            reindex(index=self.nav.index)
        # 将nan持仓填成0并归一化
        self.stock_position.holding_matrix = self.stock_position.holding_matrix.fillna(0.0)
        self.stock_position.to_percentage()

        # # 计算股票的收益序列
        # # 首先算出每天的股票净值变动
        # self.daily_stock_value_change = self.stock_value_change - self.stock_value_change.shift(1)
        # # 每天的收益率, 先用当天的股票市值总值减去当天的股票净值变动, 得到当天初的股票市值
        # # 然后用当天的股票净值变动除以当天初的股票市值
        # stock_value_begin = self.stock_value - self.daily_stock_value_change
        # self.stock_return = self.daily_stock_value_change / stock_value_begin
        self.stock_return = np.log(self.stock_value/self.stock_value.shift(1))

        # 直接用算出的股票收益序列和股票持仓矩阵进行归因
        self.pa = performance_attribution(self.stock_position, self.stock_return)
        self.pa.execute_performance_attribution(foldername=self.folder_name[:-1])
        self.pa.plot_performance_attribution(foldername=self.folder_name[:-1], pdfs=self.pdfs)

        # 储存组合的平均暴露值
        self.pa.port_expo.mean().to_csv(self.full_dir + 'mean_expo.csv', na_rep='NaN',
                                        index_label='factor', encoding='GB18030')

    # 将nav的收益率对因子收益进行回归
    def do_ts_reg(self):
        # 算净值的收益率
        self.nav_return = np.log(self.nav/self.nav.shift(1)).dropna()
        # 取barra因子的收益率, 只取与nav收益的日期相同的那一部分
        factor_return = self.pa.pa_returns.reindex(index=self.nav_return.index)

        # 读取债券和商品指数数据, 计算收益率, 也用这些进行回归
        bonds_index = pd.read_csv('bonds_index.csv', index_col=0, parse_dates=True, header=None)
        cf_index= pd.read_csv('cf_index.csv', index_col=0, parse_dates=True, header=None)
        bi_return = np.log(bonds_index/bonds_index.shift(1)).reindex(index=self.nav_return.index)
        cfi_return = np.log(cf_index/cf_index.shift(1)).reindex(index=self.nav_return.index)
        # 将这两个的return与因子return衔接起来
        # factor_return = pd.concat([factor_return, bi_return, cfi_return], axis=1)
        factor_return = pd.concat([factor_return['country_factor'], bi_return.iloc[:, 0]], axis=1)

        # 进行回归
        y = self.nav_return * 1
        x = sm.add_constant(factor_return)
        # x = factor_return
        model = sm.OLS(y, x, missing='drop')
        results = model.fit()
        # 储存回归系数
        results.params.to_csv(self.full_dir + 'ts_reg.csv', index_label='factor', na_rep='NaN',
                              encoding='GB18030')
        # 储存回归pvalues
        results.pvalues.to_csv(self.full_dir + 'ts_reg_p.csv', index_label='factor', na_rep='NaN',
                              encoding='GB18030')
        pass


    # 执行对净值表的分析
    def execute_analysis(self, *, dir_name):
        self.read_xlsx(dir_name=dir_name)
        self.handle_nav()
        self.handle_stock_position()
        self.do_ts_reg()
        self.pdfs.close()

# 主动类产品
class fof_active(fof_handler):
    # 从目录中读取xlsx文件
    def read_xlsx(self, *, dir_name):
        self.folder_name = dir_name
        # 找到指定目录下的所有excel文件
        curr_dir = os.getcwd()
        self.full_dir = curr_dir + dir_name
        files = os.listdir(self.full_dir)
        xls = [f for f in files if f[-3:] == 'xls']

        # 循环处理每一个excel文件
        for f in xls:
            self.handle_xls(self.full_dir + f)

# 量化类产品
class fof_quant(fof_handler):
    # 从包括子目录的整个目录中读取xlsx文件
    def read_xlsx(self, *, dir_name):
        self.folder_name = dir_name
        # 找到指定目录下的所有excel文件
        curr_dir = os.getcwd()
        self.full_dir = curr_dir + dir_name
        self.path = []
        self.subdirs = []
        self.files = []
        for path, subdirs, files in os.walk(self.full_dir):
            self.path.append(path)
            self.subdirs.append(subdirs)
            self.files.append(files)

        # 首先去掉前3个子目录, 并去掉空子目录, 留下代表日期的子目录
        self.subdirs = self.subdirs[3:]
        self.subdirs = [sd for sd in self.subdirs if len(sd)>0]
        # 从子目录中提取日期信息
        self.dates = []
        for i in self.subdirs:
            for item in i:
                self.dates.append(item)

        # 提取文件信息, 同步提取满足条件的文件的路径名称
        self.f = []
        self.p = []
        temp_f = [f for f, p in zip(self.files, self.path) if len(f)>0]
        temp_p = [p for f, p in zip(self.files, self.path) if len(f)>0]
        for i, j in zip(temp_f, temp_p):
            for item in i:
                if item[-3:] == 'xls':
                    self.f.append(item)
                    self.p.append(j)

        assert len(self.f) == len(self.dates), 'Error: Number of files does NOT match number of dates'
        # 循环处理文件信息
        for cursor in np.arange(len(self.f)):
            self.handle_xls(cursor)

        pass

# 处理双赢产品净值表的类
class fof_sy(fof_active):
    # 处理excel文件
    def handle_xls(self, f):
        # 取当前的估值表的日期, 当前基金净值, 基金股票价值, 每天的累计估值增值, 以及当前股票持仓
        curr_date = pd.Timestamp(f[-12:-4])
        curr_df = pd.read_excel(f, header=6)
        curr_nav = pd.Series(curr_df.iloc[-1, 11], index=[curr_date])
        curr_stock_value = pd.Series(curr_df.iloc[5, 10], index=[curr_date])

        curr_stock_position_data = curr_df[curr_df.iloc[:, 0] == '1101.01.01']
        curr_stock_position_data = curr_stock_position_data.iloc[1:, :]
        curr_stock_value_change = pd.Series(curr_stock_position_data.iloc[:, 13].sum(), index=[curr_date])
        curr_stock_index = []
        for _, stock_code in curr_stock_position_data.iloc[:, 2].iteritems():
            curr_stock_index.append(str(int(stock_code)).zfill(6))
        curr_stock_position_series = pd.Series(curr_stock_position_data.iloc[:, 10].values,
                                               index=curr_stock_index,
                                               name=curr_date)

        # 衔接数据
        self.nav = self.nav.append(curr_nav)
        self.stock_value = self.stock_value.append(curr_stock_value)
        self.stock_value_change = self.stock_value_change.append(curr_stock_value_change)
        self.stock_position.holding_matrix = self.stock_position.holding_matrix.\
            append(curr_stock_position_series)
        pass

# 处理量道的产品
class fof_ld(fof_quant):
    # 处理excel文件
    def handle_xls(self, cursor):
        # 读取当前的文件
        curr_date = pd.Timestamp(self.dates[cursor])
        curr_path = self.p[cursor]
        curr_file = self.f[cursor]

        # 取当前的净值, 股票持仓, 股票总市值等数据
        curr_df = pd.read_excel(curr_path+'/'+curr_file, header=3)
        curr_nav = pd.Series(curr_df.iloc[-5, 1], index=[curr_date])
        curr_stock_value = pd.Series(curr_df.iloc[9, 7], index=[curr_date])

        # 以1102开头的string即是股票数据, 大于8排除那些标题的行
        stock_condition = curr_df.iloc[:, 0].apply(lambda x:str(x).startswith('1102') and len(str(x))>8)
        curr_stock_data = curr_df[stock_condition]
        # 取股票的代码, 是科目代码的后6位
        curr_stock_index = []
        for _, code in curr_stock_data.iloc[:, 0].iteritems():
            stock_code = str(code)[-6:]
            curr_stock_index.append(stock_code)
        curr_stock_position_series = pd.Series(curr_stock_data.iloc[:, 7].values,
                                               index=curr_stock_index,
                                               name=curr_date)

        # 衔接数据
        self.nav = self.nav.append(curr_nav)
        self.stock_value = self.stock_value.append(curr_stock_value)
        self.stock_position.holding_matrix = self.stock_position.holding_matrix.\
            append(curr_stock_position_series)

        pass

# 处理龙旗的产品
class fof_lq(fof_quant):
    def handle_xls(self, cursor):
        # 读取当前的文件
        curr_date = pd.Timestamp(self.dates[cursor])
        curr_path = self.p[cursor]
        curr_file = self.f[cursor]

        # 取当前的净值, 股票持仓, 股票总市值等数据
        curr_df = pd.read_excel(curr_path + '/' + curr_file, header=4)
        print(curr_date)
        if curr_date in [pd.Timestamp('2017-03-24'), pd.Timestamp('2017-03-27'),pd.Timestamp('2017-03-28'),
                         pd.Timestamp('2017-03-29')]:
            curr_nav = pd.Series(float(curr_df.iloc[-9, 1]), index=[curr_date])
        else:
            curr_nav = pd.Series(float(curr_df.iloc[-8, 1]), index=[curr_date])
        curr_stock_value = pd.Series(curr_df.iloc[12, 7], index=[curr_date])

        # 以1102开头的string即是股票数据, 大于10排除那些标题的行
        stock_condition = curr_df.iloc[:, 0].apply(lambda x: str(x).startswith('1102') and len(str(x)) > 10)
        curr_stock_data = curr_df[stock_condition]
        # 取股票的代码, 是科目代码的倒数第3-9位
        curr_stock_index = []
        for _, code in curr_stock_data.iloc[:, 0].iteritems():
            stock_code = str(code)[-9:-3]
            curr_stock_index.append(stock_code)
        curr_stock_position_series = pd.Series(curr_stock_data.iloc[:, 7].values,
                                               index=curr_stock_index,
                                               name=curr_date)

        # 衔接数据
        self.nav = self.nav.append(curr_nav)
        self.stock_value = self.stock_value.append(curr_stock_value)
        self.stock_position.holding_matrix = self.stock_position.holding_matrix. \
            append(curr_stock_position_series)
        pass

# 处理天演的产品
class fof_ty(fof_quant):
    # 处理excel文件
    def handle_xls(self, cursor):
        # 读取当前的文件
        curr_date = pd.Timestamp(self.dates[cursor])
        curr_path = self.p[cursor]
        curr_file = self.f[cursor]

        # 取当前的净值, 股票持仓, 股票总市值等数据
        curr_df = pd.read_excel(curr_path + '/' + curr_file, header=3)
        curr_nav = pd.Series(curr_df.iloc[-5, 1], index=[curr_date])
        curr_stock_value = pd.Series(curr_df.iloc[9, 7], index=[curr_date])

        # 以1102开头的string即是股票数据, 大于8排除那些标题的行
        stock_condition = curr_df.iloc[:, 0].apply(lambda x: str(x).startswith('1102') and len(str(x)) > 8)
        curr_stock_data = curr_df[stock_condition]
        # 取股票的代码, 是科目代码的后6位
        curr_stock_index = []
        for _, code in curr_stock_data.iloc[:, 0].iteritems():
            stock_code = str(code)[-6:]
            curr_stock_index.append(stock_code)
        curr_stock_position_series = pd.Series(curr_stock_data.iloc[:, 7].values,
                                               index=curr_stock_index,
                                               name=curr_date)

        # 衔接数据
        self.nav = self.nav.append(curr_nav)
        self.stock_value = self.stock_value.append(curr_stock_value)
        self.stock_position.holding_matrix = self.stock_position.holding_matrix. \
            append(curr_stock_position_series)

        pass

# 处理博普套利的产品
class fof_bptl(fof_quant):
    def handle_xls(self, cursor):
        # 读取当前的文件
        curr_date = pd.Timestamp(self.dates[cursor])
        curr_path = self.p[cursor]
        curr_file = self.f[cursor]

        # 取当前的净值, 股票持仓, 股票总市值等数据
        curr_df = pd.read_excel(curr_path + '/' + curr_file, header=4)
        print(curr_date)
        if curr_date in [pd.Timestamp('2017-03-24'), pd.Timestamp('2017-03-27'),pd.Timestamp('2017-03-28'),
                         pd.Timestamp('2017-03-29')]:
            curr_nav = pd.Series(float(curr_df.iloc[-9, 1]), index=[curr_date])
        else:
            curr_nav = pd.Series(float(curr_df.iloc[-8, 1]), index=[curr_date])
        if not curr_df[curr_df.iloc[:, 0]=='1102'].empty:
            curr_stock_value = pd.Series(curr_df[curr_df.iloc[:, 0]=='1102'].iloc[0, 7], index=[curr_date])
        else:
            curr_stock_value = pd.Series(0, index=[curr_date])

        # 以1102开头的string即是股票数据, 大于10排除那些标题的行
        stock_condition = curr_df.iloc[:, 0].apply(lambda x: str(x).startswith('1102') and len(str(x)) > 10)
        # 这只产品可能出现没有股票持仓的情况
        if (stock_condition == False).all():
            curr_stock_position_series = pd.Series()
        else:
            curr_stock_data = curr_df[stock_condition]
            # 取股票的代码, 是科目代码的倒数第3-9位
            curr_stock_index = []
            for _, code in curr_stock_data.iloc[:, 0].iteritems():
                stock_code = str(code)[-9:-3]
                curr_stock_index.append(stock_code)
            curr_stock_position_series = pd.Series(curr_stock_data.iloc[:, 7].values,
                                                   index=curr_stock_index,
                                                   name=curr_date)

        # 衔接数据
        self.nav = self.nav.append(curr_nav)
        self.stock_value = self.stock_value.append(curr_stock_value)
        if not curr_stock_position_series.empty:
            self.stock_position.holding_matrix = self.stock_position.holding_matrix. \
                append(curr_stock_position_series)
        pass


# 做pca的函数
def do_pca():
    # 对主动产品做
    sy7 = fof_sy()
    sy7.execute_analysis(dir_name='/主动估值表/双赢7/')
    sy10 = fof_sy()
    sy10.execute_analysis(dir_name='/主动估值表/双赢10/')
    sy11 = fof_sy()
    sy11.execute_analysis(dir_name='/主动估值表/双赢11/')

    # 粘贴它们的净值序列
    nav = pd.concat([sy7.nav, sy10.nav, sy11.nav], axis=1).fillna(method='ffill')

    # 对量化产品做
    bptl = fof_bptl()
    bptl.execute_analysis(dir_name='/量化估值表/2017博普套利/')
    ld = fof_ld()
    ld.execute_analysis(dir_name='/量化估值表/2017量道/')
    lq = fof_lq()
    lq.execute_analysis(dir_name='/量化估值表/2017龙旗/')
    ty = fof_ty()
    ty.execute_analysis(dir_name='/量化估值表/2017天演/')

    # 粘贴它们的净值序列
    nav = pd.concat([bptl.nav, ld.nav, lq.nav, ty.nav], axis=1).fillna(method='ffill')

    nav_r = np.log(nav/nav.shift(1))
    nav_r = nav_r.ix[1:, :]

    # 算相关系数矩阵
    corr_matrix = np.corrcoef(nav_r.T)

    # 做pca
    pca = PCA()
    pc_ratio = pca.fit(nav_r).explained_variance_ratio_
    print('pc_ratio: '+str(pc_ratio)+'\n')

    # 做mvo
    nav_r = np.log(nav/nav.shift(1))
    nav_r = nav_r.ix[1:, :]
    results = optimal_portfolio_with_virtual_mean(nav_r.T.values, 0)
    print('opt_result: '+str(results.x)+'\n')

    pass

def target_func(x, cov_matix, mean_vector, virtual_mean):
    f = float(-(x.dot(mean_vector) - virtual_mean) / np.sqrt(x.dot(cov_matix).dot(x.T)))
    return f


def optimal_portfolio_with_virtual_mean(profits, virtual_mean, allow_short=False):
    x = np.ones(len(profits))
    mean_vector = np.mean(profits, axis=1)
    cov_matrix = np.cov(profits)
    cons = ({'type': 'eq',
             'fun': lambda x: np.sum(x) - 1})
    if not allow_short:
        bounds = [(0.15, None,) for i in range(len(x))]
    else:
        bounds = None
    minimize = optimize.minimize(target_func, x, args=(cov_matrix, mean_vector, virtual_mean,), bounds=bounds,
                                 constraints=cons)
    return minimize


if __name__ == '__main__':
    fh = fof_sy()
    # fh = fof_ty()
    fh.execute_analysis(dir_name='/主动估值表/双赢11/')
    # do_pca()
    pass
