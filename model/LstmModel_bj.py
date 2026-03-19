import pandas as pd  # 添加导入
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.autograd import Variable
import math


class LstmDataset():
    # mydataset需要的几个函数：
    # normalization
    # split_windows
    # to tensor
    def __init__(self, sequence_length = 7, batch_size = 2, pred_stamp = 1):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.pred_stamp = pred_stamp
        self.max = {}
        self.min = {}
        self.feature_names = None  # 新增：存储特征名称

    def _init_scaler(self, df_train):
        """
        compute the max and min by using train data
        ----------------------------
        df_train: the train dataframe
        """
        # 保存特征名称
        self.feature_names = list(df_train.columns)
        
        for c in df_train.columns:
            self.max[c] = np.max(df_train[c].values)
            self.min[c] = np.min(df_train[c].values)

    def get_scaler(self):
        return self.max, self.min

    def maxmin_normalization(self, df):
        """
        ---------
        max-min normalization
        df: dataframe
        """
        data = copy.deepcopy(df)
        for c in df.columns:
            if c in self.max.keys():
                data[c] = (data[c]-self.min[c])/(self.max[c]-self.min[c])
            else:
                print("the columns ",c," does not exist!")
        data = data.astype(np.float32)
        return data.values
    
    def inverse_normalization(self, df):
        for c in df.columns:
            df[c] = df[c]*(self.max[c]-self.min[c])+self.min[c]
        return df
    
    def get_rate_scaler(self):
        return self.max['rate'], self.min['rate']
    
    def get_feature_names(self):
        return self.feature_names
    
    def split_windows(self, data):
        """
        ----------
        对序列数据按照时间窗划分
        输入data为多维数组
        """
        x=[]
        y=[]
        for i in range(len(data)-self.sequence_length-self.pred_stamp+1): # range的范围需要减去时间步长和1
            _x = data[i:(i+self.sequence_length),:] # (sequence_length, input_dim)
            _y = data[(i+self.sequence_length):(i+self.sequence_length+self.pred_stamp),-1] # (pred_stamp, 1)
            x.append(_x)
            y.append(_y)
        x, y = np.array(x), np.array(y)
        return x, y
    
    def to_tensor(self, data):
        """
        from ndarray to torch.tensor 
        ------------------------------
        data : nd-array
        """
        return Variable(torch.Tensor(np.array(data)))

    def get_train_val_test_dataset(self, df, test_size = 0, sample_rate = None, train_pure_data = False, validation = True):
        """
        get the train, validation and test data from df dataframe
        ----------------------------------
        df : DataFrame
        test_size: the n size of test data
        """
        dataset = copy.deepcopy(df).values
        n = df.shape[0]
        self._init_scaler(df_train = df.iloc[0:(n-test_size),:])
        train_dataset = self.maxmin_normalization(df.iloc[0:(n-test_size),:])
        test_dataset = self.maxmin_normalization(df.iloc[(n-test_size-self.sequence_length):,:])
        # split window
        x_data, y_data = self.split_windows(train_dataset)
        x_test, y_test = self.split_windows(test_dataset)
        # split train and validation data
        if sample_rate is not None:
            data_ind = np.random.choice(x_data.shape[0], size=int(x_data.shape[0]*sample_rate), replace=True)
            x_data_sample, y_data_sample = x_data[data_ind,:], y_data[data_ind]
        else:
            x_data_sample, y_data_sample = copy.deepcopy(x_data), copy.deepcopy(y_data)
        if validation == True:
            val_size = math.ceil(x_data_sample.shape[0] * 0.9)
            x_train, y_train = x_data_sample[0:val_size,:,:],y_data_sample[0:val_size]
            x_val, y_val = x_data_sample[val_size:,:,:], y_data_sample[val_size:]
        else:
            x_train, y_train = x_data_sample,y_data_sample
            x_val, y_val = x_data_sample, y_data_sample
        # turn to tensor 
        x_data, y_data = self.to_tensor(x_data), self.to_tensor(y_data)
        x_train, y_train = self.to_tensor(x_train), self.to_tensor(y_train)
        x_val, y_val = self.to_tensor(x_val), self.to_tensor(y_val)
        x_test, y_test = self.to_tensor(x_test), self.to_tensor(y_test)
        train_dataset = TensorDataset(x_train, y_train)
        if self.batch_size > 0:
            train_dataloader = DataLoader(dataset=train_dataset,batch_size=self.batch_size, shuffle=True, drop_last=False)
            val_datadict = {'x_data':x_val, 'y_data':y_val}
            test_datadict = {'x_data':x_test, 'y_data':y_test}
            if train_pure_data is True:
                train_datadict = {'x_data':x_data,'y_data':y_data}
                return train_dataloader, val_datadict, test_datadict, train_datadict
            return train_dataloader, val_datadict, test_datadict
        else:
            train_datadict = {'x_data':x_train, 'y_data':y_train}
            val_datadict = {'x_data':x_val, 'y_data':y_val}
            test_datadict = {'x_data':x_test, 'y_data':y_test}
            return train_datadict, val_datadict, test_datadict