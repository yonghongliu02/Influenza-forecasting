"""
Direct multi-horizon LSTM
Yearly rolling
Point forecasting only
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import torch
import copy
import os
import sys
sys.path.append(".")

from datetime import datetime, timedelta
from model.LstmModel import LstmDataset, LstmModel, LstmTrain
from tools.data import DataTool

###############################################
# 0. 固定随机性（点估计非常重要）
###############################################
SEED = 2023
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############################################
# 1. 基本配置
###############################################
model_name = 'LSTM_direct_multioutput_rolling'
mode = 'test8'

seq_length = 14
pred_horizon = 9

mid_dim = 128
hidden_layers = 2
dropout_rate = 0.3
lr = 0.01
batch_size = 8
num_epochs = 500
early_stopping = 20

###############################################
# 2. 数据读取
###############################################
pwd = os.path.abspath(os.path.dirname(os.getcwd()))
path = pwd + '/Data/deep_learn_data1022.csv'

col_list = [
    'mean_temperature',
    'rh',
    'monthid',
    'weekid',
    'rate'
]

dr = DataTool()
df_o = dr.data_output(path, col_list, mode='log')
df = copy.deepcopy(df_o)

###############################################
# 3. 单次 rolling 预测
###############################################
def one_rolling(df, test_start, test_end, pred_horizon, exp_mode=True):

    # 只使用 test_end 之前的数据
    df_t = copy.deepcopy(df.loc[df.index <= pd.to_datetime(test_end), :])

    # 测试集长度
    test_size = df_t.loc[df_t.index > test_start, :].shape[0]

    # 真实值（用于评估）
    df_test = copy.deepcopy(df_t.loc[df_t.index >= test_start, :])
    re_test = dr.origin_re_output(
        df_test,
        left_len=0,
        pred_len=pred_horizon,
        exp_mode=exp_mode
    )

    ###########################################
    # Dataset
    ###########################################
    data_deal = LstmDataset(
        sequence_length=seq_length,
        batch_size=batch_size,
        pred_stamp=pred_horizon
    )

    train_dataloader, val_datadict, test_datadict = \
        data_deal.get_train_val_test_dataset(
            copy.deepcopy(df_t),
            test_size=test_size,
            sample_rate=None
        )

    rate_max, rate_min = data_deal.get_rate_scaler()

    x_test = test_datadict['x_data']
    y_test = test_datadict['y_data']

    ###########################################
    # Model
    ###########################################
    input_dim = df.shape[1]
    output_dim = pred_horizon

    model = LstmModel(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_dim=seq_length,
        mid_dim=mid_dim,
        hidden_lstm_layers=hidden_layers,
        dropout_rate=dropout_rate,
        num_directions=1,
        MCDropout=None
    )

    model_train = LstmTrain(model)

    model_train.train(
        train_dataloader,
        val_datadict,
        num_epochs=num_epochs,
        lr=lr,
        early_stopping=early_stopping,
        verboose=2
    )

    ###########################################
    # Predict
    ###########################################
    _, y_test_pred = model_train.predict_xy(x_test, y_test)
    # shape: [n_sample, pred_horizon]

    # 反归一化 + exp
    y_test_pred = y_test_pred * (rate_max - rate_min) + rate_min
    y_test_pred = np.exp(y_test_pred)

    ###########################################
    # 组织输出
    ###########################################
    re_test_pred = []
    for k in range(pred_horizon):
        tmp = pd.DataFrame(
            y_test_pred[:, k],
            columns=['point']
        )
        tmp['week_ahead'] = k
        re_test_pred.append(tmp)

    re_test_pred = pd.concat(re_test_pred, ignore_index=True)
    re_test = pd.concat([re_test, re_test_pred[['point']]], axis=1)

    return re_test


###############################################
# 4. Yearly rolling
###############################################
test_start_date = pd.to_datetime('2015-07-06')

rolling_dates = [
    test_start_date + timedelta(days=52 * 7 * i)
    for i in range(20)
    if test_start_date + timedelta(days=52 * 7 * i)
    < (pd.to_datetime('2024-08-27') - timedelta(days=(pred_horizon - 1) * 7))
]

rolling_dates.append(
    pd.to_datetime('2024-08-27') - timedelta(days=(pred_horizon - 1) * 7)
)

re_test_total = pd.DataFrame()

for i in range(len(rolling_dates) - 1):
    test_start = rolling_dates[i]
    test_end = rolling_dates[i + 1] + timedelta(days=(pred_horizon - 1) * 7)

    print(
        f'Rolling {i}: test_start={test_start}, test_end={test_end}'
    )

    re_test = one_rolling(
        df,
        test_start,
        test_end,
        pred_horizon
    )

    re_test_total = pd.concat([re_test_total, re_test], axis=0)

###############################################
# 5. 后处理 & 保存
###############################################
re_test_total['point'] = np.exp(re_test_total['point']) / 10
re_test_total['true'] = np.exp(re_test_total['true']) / 10
re_test_total['point_avg']=re_test_total['point']
dr.point_write(
    re=re_test_total,
    origin_path=pwd,
    mode=mode,
    model_name=model_name
)

print("Finished:", model_name)
