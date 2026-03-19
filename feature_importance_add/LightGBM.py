"""
LIghtGBM model train and predict
Yearly rolling 

"""

## import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import xgboost
warnings.filterwarnings('ignore')
import random
import copy
import os
import sys
import lightgbm as lgb
from collections import defaultdict  # 新增导入
# 获取当前工作目录
current_dir = os.getcwd()
# 获取上一层目录
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import datetime,timedelta
from model.MLModel_bj import LightGBMmodel, MLDataset  # 修改导入路径
from tools.data import DataTool
from tools.plot import Plot_
############################################### preparation ##########################################
model_name = 'LGBM_rolling'
mode = 'test8_2023_add'
#test_start_date = '2005-11-01'
#test_end_date = None#'2016-06-30'

start_time = datetime.now()

#data路径
pwd=os.path.abspath(os.path.dirname(os.getcwd()))
origin_path=os.path.abspath(os.path.join(pwd, '..'))
path = pwd + '/Data/rolling_data_before1022.parquet'

##预测变量
col_list = ['mean_temperature','rh', "absenteeism",
                      'monthid', 'weekid', 'rate', 'date_analysis']
dr = DataTool()
df_o = dr.data_output(path, col_list, mode = 'log')

#转换为时间格式
df_o['date_analysis'] = pd.to_datetime(df_o['date_analysis'])
#赋值数据集
df = copy.deepcopy(df_o)
print("finally, start date = ", df.index.min())
# 筛选出最新日期的行，并删除 'date_analysis' 列
df_all = df.loc[df['date_analysis'] == max(df['date_analysis']),].drop('date_analysis', axis = 1)

############################################### fit and predict ##########################################
pred_stamp = 9
exp_mode = True
cov_list = ['mean_temperature','rh',"absenteeism"]

def cv_param(df, random_state, test_start, pred_horizon):
    """ 
    first, update the parameter
    second, fit model and make prediction
    -----------------------------------------
    """
    #设置种子
    np.random.seed(random_state)
    random.seed(random_state)
    
    #找到训练数据的结束日期
    train_analysis_end = max([date_ for date_ in df.date_analysis.unique() 
                              if date_ <= test_start])
    
    #从 df 中选择出日期为 train_analysis_end 的行，作为训练集。并删除 date_analysis 列
    df_train = copy.deepcopy(df.loc[df.date_analysis == train_analysis_end,:])
    df_train = df_train.drop('date_analysis', axis = 1)
    
    mylgb = LightGBMmodel()
    data_deal = MLDataset()
    # 1. -------------------- get the best parameter ------------
    ### train data prepare
    train_datadict, max_lag, feature_names = data_deal.get_train_data(df_train, 
                                                 max_rate_lag=14,
                                                 cov_list = cov_list, 
                                                 max_cov_lag=14, 
                                                 pred_horizon = pred_horizon, 
                                                 validation=False,
                                                 return_feature_names=True)  # 修改为返回特征名称
    x_train, y_train = train_datadict['x_data'], train_datadict['y_data']
    
    # LightGBM模型的超参数网格
    params = {
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [300, 600],
        'max_depth': [2, 3, 4],
        'num_leaves': [8, 16, 31],
        'random_state': [random_state],
        'verbose': [-1]
    }
 
    # 直接调用CV_train_，它会处理DataFrame到numpy的转换
    mylgb.CV_train_(x_train, y_train, fold_num = 3, param_dict = params)
  
    return mylgb, data_deal, train_datadict, feature_names  # 返回特征名称

def fit_and_predict(df, model, data_deal, train_datadict, pred_start, pred_end, pred_horizon, random_state, feature_names):
    #设种子
    np.random.seed(random_state)
    random.seed(random_state)
    
    # re fit with sampling 
    x_train, y_train = train_datadict['x_data'].values, train_datadict['y_data'].values
    train_ind = np.random.choice(x_train.shape[0], size=x_train.shape[0], replace=True)
    x_train,y_train = x_train[train_ind,:],y_train[train_ind,:]
    model.fit_(x_train, y_train, random_state)
    model.feature_names = feature_names  # 确保模型保存特征名称
    
    # 获取特征重要性
    importance_array, feature_names = model.get_feature_importance()
    print("Feature importance calculated for LightGBM model")
    
    # 为每个预测周期创建重要性DataFrame
    importance_dfs = []
    for week_ahead in range(pred_horizon):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_array,
            'week_ahead': week_ahead
        })
        importance_dfs.append(importance_df)
    
    # 合并所有预测周期的重要性
    all_importance_df = pd.concat(importance_dfs, ignore_index=True)
   
    # test data preparation
    max_lag = data_deal.output_max_lag()
    test_start_date = pred_start+timedelta(days=-int(7*max_lag))
    test_end_date = pred_end
    
    #预测区间
    df_test_o = df.loc[(df.index >= test_start_date) & (df.index <= test_end_date),:]
    df_test= data_deal.get_test_data(df_test_o)
    x_test, y_test = df_test.iloc[:,0:-pred_horizon].values, df_test.iloc[:,-pred_horizon:].values
    
    # make prediction
    y_test_hat = model.predict_(x_test)
    
    #反归一化处理
    rate_max, rate_min = data_deal.output_rate_scaler()
    y_test, y_test_hat = y_test*(rate_max - rate_min)+rate_min , y_test_hat*(rate_max - rate_min)+rate_min 
    y_test, y_test_hat = np.exp(y_test), np.exp(y_test_hat)
    
    return y_test, y_test_hat, all_importance_df


#生成滚动日期
test_start_date = pd.to_datetime('2023-07-03')
max_year_range = 15
year_step = 1
rolling_dates = [test_start_date + 
                 timedelta(days = 52 * 7 * i) 
                 for i in range(0,max_year_range,year_step) 
                 if test_start_date + timedelta(days = 52 * 7 * i) < (pd.to_datetime('2025-07-03')-timedelta(days = (pred_stamp-1) * 7))]

#添加最终的滚动日期
rolling_dates.append(pd.to_datetime('2025-07-03')-timedelta(days = (pred_stamp-1) * 7))

#准备测试数据
df_test_total = copy.deepcopy(df.loc[df.index >= test_start_date,:])
#空数据框
re_test_total = pd.DataFrame()
feature_importance_results = defaultdict(list)  # 新增：用于存储特征重要性结果

bootstrap_times = 1
for i_date in range(len(rolling_dates)-1):
    print("this prediction interval = ", rolling_dates[i_date], rolling_dates[i_date+1])
    test_start, test_end = rolling_dates[i_date], rolling_dates[i_date+1]+timedelta(days = (pred_stamp-1) * 7)
    #数据准备
    df_t = copy.deepcopy(df_all.loc[df_all.index<=pd.to_datetime(test_end),:])
    df_test = copy.deepcopy(df_t.loc[df_t.index >= test_start,:])
    #生成预测数据并调用模型函数
    re_test = dr.origin_re_output(df_test, left_len=0, pred_len = pred_stamp, exp_mode=exp_mode)
    print("----------------------------- i_date = ", i_date,", test_start = ", test_start, ', test_end = ',test_end)
    
    #使用交叉验证和参数调整
    mymodel, data_deal, train_datadict, feature_names = cv_param(df, random_state = i_date, test_start = test_start, pred_horizon = pred_stamp)
    
    #使用 Bootstrap 重采样
    for bst in range(bootstrap_times):
        print("-------------------------- bootstrap = ", bst, " ----------------------------------")
        y_true, y_pred, importance_df = fit_and_predict(df_all, mymodel, data_deal, train_datadict, 
                                                       test_start, test_end, pred_stamp, 
                                                       random_state=i_date,
                                                       feature_names=feature_names)
        
        # 保存特征重要性结果
        importance_df['rolling_window'] = i_date
        importance_df['test_start'] = test_start
        importance_df['test_end'] = test_end
        importance_df['bootstrap'] = bst
        feature_importance_results[f'window_{i_date}_bst_{bst}'] = importance_df
        
        re_test_pred = pd.DataFrame()
        for i in range(pred_stamp):
            re_t = pd.DataFrame(y_pred[:,i], columns = [f'boot_{bst}'])
            re_t['week_ahead'] = i
            re_test_pred = pd.concat([re_test_pred, re_t], ignore_index=True)
        re_test = pd.concat([re_test, re_test_pred[[f'boot_{bst}']]], axis=1)
    
    #汇总所有的测试结果
    re_test_total = pd.concat([re_test_total, re_test], axis=0)

############################################### save ##########################################
# 保存特征重要性结果
fi_path = pwd + f'/Results/FI_add/fi_{model_name}_{mode}.csv'
all_importance_dfs = pd.concat(feature_importance_results.values(), ignore_index=True)
all_importance_dfs.to_csv(fi_path, index=False)

# 保存预测结果
re_test_total.rename(columns={'boot_0':'point'}, inplace=True)
#转换
re_test_total['point'] = np.exp(re_test_total['point']) / 10
re_test_total['true'] = np.exp(re_test_total['true']) / 10
re_test_total['point_avg'] = re_test_total['point']

dr.point_write(re = re_test_total, origin_path=pwd, mode = mode, model_name=model_name)

end_time = datetime.now()
print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name,
      " begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
print("The running time totally =", (end_time-start_time).seconds," seconds.")