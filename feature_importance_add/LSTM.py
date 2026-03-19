"""
Direct multi-horizon LSTM
Yearly rolling
Point forecasting with feature importance (saliency)
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
mode = 'test8_2023_add'

seq_length = 14
pred_horizon = 9

mid_dim = 128
hidden_layers = 2
dropout_rate = 0.3
lr = 0.003
batch_size = 8
num_epochs = 500
early_stopping = 100

###############################################
# 2. 数据读取
###############################################
pwd = os.path.abspath(os.path.dirname(os.getcwd()))
path = pwd + '/Data/deep_learn_data1022.csv'

col_list = [
    'mean_temperature',
    'rh',
    "absenteeism",
    'monthid',
    'weekid',
    'rate'
]

dr = DataTool()
df_o = dr.data_output(path, col_list, mode='log')
df = copy.deepcopy(df_o)

###############################################
# 3. 单次 rolling 预测及变量重要性计算
###############################################
def one_rolling(df, test_start, test_end, pred_horizon, exp_mode=True):
    """
    执行单次rolling训练、预测和变量重要性计算
    """
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
    # 获取训练后的模型
    ###########################################
    model_after = model_train.output_model()
    
    ###########################################
    # Predict
    ###########################################
    _, y_test_pred = model_train.predict_xy(x_test, y_test)
    # shape: [n_sample, pred_horizon]
    
    # 反归一化 + exp
    y_test_pred_original = y_test_pred * (rate_max - rate_min) + rate_min
    y_test_pred_original = np.exp(y_test_pred_original)
    
    ###########################################
    # 变量重要性计算 (Saliency Map)
    ###########################################
    # 确保模型在评估模式
    model_after.eval()
    
    # 确保x_test需要梯度
    x_test.requires_grad_()
    
    # 前向传播
    predictions = model_after.forward(x_test)
    
    # 为每个预测步长计算saliency
    feature_names = [f'{col}_{ii}d' for col in col_list for ii in range(seq_length, 0, -1)]
    saliency_df = pd.DataFrame()
    saliency_df['feature_name'] = feature_names
    
    for horizon_idx in range(pred_horizon):
        # 计算当前预测步长的saliency
        saliency = None
        
        # 获取当前步长的预测结果并求和（用于计算梯度）
        # 使用所有样本的当前步长预测值之和
        pred_sum = predictions[:, horizon_idx].sum()
        
        # 反向传播
        pred_sum.backward(retain_graph=(horizon_idx < pred_horizon - 1))
        
        # 获取梯度绝对值
        saliency_tmp = abs(x_test.grad.data)
        
        # 对序列长度和特征维度进行展开
        saliency_flat = torch.flatten(saliency_tmp, start_dim=1, end_dim=2)
        
        # 对该rolling窗口的所有样本求和（参考第一个文件的做法）
        # 注意：第一个文件中取了 n = saliency.shape[0] - total_pred_horizon - 1
        # 这里我们保持类似逻辑，但根据实际情况调整
        n_samples = saliency_flat.shape[0]
        # 去掉最后total_pred_horizon个样本（避免边界效应）
        valid_samples = max(0, n_samples - pred_horizon - 1)
        if valid_samples > 0:
            saliency_sum = saliency_flat[:valid_samples, :].sum(dim=0)
        else:
            saliency_sum = saliency_flat.sum(dim=0)
        
        # 保存到DataFrame
        saliency_df[f'week{horizon_idx}'] = saliency_sum.detach().numpy()
        
        # 清零梯度以备下一次计算
        if x_test.grad is not None:
            x_test.grad.zero_()
    
    # 添加时间信息
    saliency_df['date'] = test_start
    
    ###########################################
    # 组织预测结果输出
    ###########################################
    re_test_pred = []
    for k in range(pred_horizon):
        tmp = pd.DataFrame(
            y_test_pred_original[:, k],
            columns=['point']
        )
        tmp['week_ahead'] = k
        re_test_pred.append(tmp)
    
    re_test_pred = pd.concat(re_test_pred, ignore_index=True)
    re_test = pd.concat([re_test, re_test_pred[['point']]], axis=1)
    
    return re_test, saliency_df


###############################################
# 4. Yearly rolling
###############################################
test_start_date = pd.to_datetime('2023-07-03')

rolling_dates = [
    test_start_date + timedelta(days=52 * 7 * i)
    for i in range(20)
    if test_start_date + timedelta(days=52 * 7 * i)
    < (pd.to_datetime('2025-07-03') - timedelta(days=(pred_horizon - 1) * 7))
]

rolling_dates.append(
    pd.to_datetime('2025-07-03') - timedelta(days=(pred_horizon - 1) * 7)
)

re_test_total = pd.DataFrame()
saliency_total = pd.DataFrame()

for i in range(len(rolling_dates) - 1):
    test_start = rolling_dates[i]
    test_end = rolling_dates[i + 1] + timedelta(days=(pred_horizon - 1) * 7)
    
    print(
        f'Rolling {i}: test_start={test_start}, test_end={test_end}'
    )
    
    re_test, saliency_df = one_rolling(
        df,
        test_start,
        test_end,
        pred_horizon
    )
    
    re_test_total = pd.concat([re_test_total, re_test], axis=0)
    saliency_total = pd.concat([saliency_total, saliency_df], axis=0)

###############################################
# 5. 后处理 & 保存
###############################################
# 预测结果后处理
re_test_total['point'] = np.exp(re_test_total['point']) / 10
re_test_total['true'] = np.exp(re_test_total['true']) / 10
re_test_total['point_avg'] = re_test_total['point']

# 保存预测结果
dr.point_write(
    re=re_test_total,
    origin_path=pwd,
    mode=mode,
    model_name=model_name
)

# 保存变量重要性结果
fi_path = pwd + f'/Results/FI_add/fi_{model_name}_{mode}.csv'
saliency_total.to_csv(fi_path, index=False)

print("预测结果和变量重要性计算完成！")
print(f"预测结果已保存，变量重要性结果已保存至: {fi_path}")