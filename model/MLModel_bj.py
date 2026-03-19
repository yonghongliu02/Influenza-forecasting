import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# from statsmodels.tsa.arima_model import ARIMA
# from pmdarima import auto_arima
import copy
import math
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import Lasso, LassoCV


class MLDataset():
    # mydataset需要的几个函数：
    # normalization
    # split_windows
    # to tensor
    def __init__(self):
        self.max = {}
        self.min = {}
        self.lag_mode = None
        self.rate_lag = None
        self.cov_list = None
        self.max_test_lag = None
        self.cov_lagdict = None
        self.max_lag_order = 0
        self.pred_horizon = 0

    def _init_scaler(self, df_train):
        """
        compute the max and min by using train data
        ----------------------------
        df_train: the train dataframe
        """
        for c in df_train.columns:
            self.max[c] = np.max(df_train[c].values)
            self.min[c] = np.min(df_train[c].values)
        # print(self.max)
        # print(self.min)

    def output_scaler(self):
        return self.max, self.min

    def maxmin_normalization(self, df):
        """
        max-min normalization，对数据进行标准化处理
        ---------
        df: dataframe
        """
        data = copy.deepcopy(df)
        for c in df.columns:
            if c in self.max.keys():
                data[c] = (data[c]-self.min[c])/(self.max[c]-self.min[c])
            else:
                print("the columns ",c," does not exist!")
        data = data.astype(np.float64)
        return data
    
    def inverse_normalization(self, df):
        for c in df.columns:
            df[c] = df[c]*(self.max[c]-self.min[c])+self.min[c]
        return df
    
    def output_rate_scaler(self):
        """
        get the max and min scaler of rate variable
        ----------------
        """
        return self.max['rate'], self.min['rate']

    def _get_cov_best_lag(self, df):
        col_origin = list(df.columns)
        df_cor = pd.DataFrame(columns=['var','lag','pearson'])
        ### 计算各阶自变量滞后的相关系数结果
        for i in range(len(self.cov_list)):
            df_i = df[['rate', self.cov_list[i]]]
            # if i == 0:
            #     print(df_i)
            for lag in range(1, self.max_cov_lag+1):
                df_i[self.cov_list[i]] = df_i[self.cov_list[i]].shift(lag) #shift(正值)表示向下shift，以前的shift到现在
                tmp_cor = df_i.corr()
                # print(cov_list[i], tmp_cor)
                df_cor.loc[len(df_cor.index),:] = np.array([self.cov_list[i], lag, tmp_cor.loc['rate',self.cov_list[i]]])

        df_cor['pearson'] = df_cor['pearson'].astype('float64').abs()
        df_cor['lag'] = df_cor['lag'].astype('int')
        idx = df_cor.groupby('var')['pearson'].idxmax()
        df_lag = df_cor.iloc[idx,:][['var', 'lag']]
        # print("--- df_lag-------")
        # print(df_lag)
        cov_lagdict = dict()
        for i in range(df_lag.shape[0]):
            # print("best lag : ", df_lag.iloc[i,:]['var'], df_lag.iloc[i,:]['lag'])
            cov_lagdict[df_lag.iloc[i,:]['var']] = df_lag.iloc[i,:]['lag']
        
        ## 计算各阶rate滞后的相关系数结果
        df_cor_rate = pd.DataFrame(columns=['var','lag','pearson'])
        for l in range(1, self.max_rate_lag+1):
            df_i = copy.deepcopy(df[['rate']])
            df_i['rate_lagged'] = df_i['rate'].shift(l)
            tmp_cor = df_i.corr()
            df_cor_rate.loc[len(df_cor_rate.index),:] = np.array(['rate', l, tmp_cor.loc['rate','rate_lagged']])
        df_cor_rate['pearson'] = df_cor_rate['pearson'].astype('float64').abs()
        df_cor_rate['lag'] = df_cor_rate['lag'].astype('int')
        df_cor_rate = df_cor_rate.loc[df_cor_rate.pearson > 0.5,:]
        rate_lag = max(df_cor_rate.lag)
        cov_lagdict['rate'] = rate_lag
        # 得到最终的最优的滞后
        self.cov_lagdict = cov_lagdict
        print("the lags having best correlation for covariates : ", self.cov_lagdict)
              
    def output_best_lag(self):
        return self.cov_lagdict
    
    def output_max_lag(self):
        return self.max_lag_order
    
    def _deal_cov_lag_train(self, df):
        """
        auto choose lag order by using correlation method, and then finish lag
        -----------------------------
        df : DataFrame
        cov_list: the covriate list
        """
        self._get_cov_best_lag(df = copy.deepcopy(df))

        df_here = copy.deepcopy(df)
        max_lag = 0
        for k in self.cov_lagdict.keys():
            if k not in df_here.columns:
                print(k," is not a column of data!!!")
            else:
                lag_num =  self.cov_lagdict[k]
                # print(k," lag ",lag_num," order")
                max_lag = max(max_lag, lag_num)
                for l in range(1, lag_num+1):
                    df_here[f'{k}_{l}d'] = df_here[k].shift(l) 

        self.max_lag_order = max_lag
        # col_drop = list(set(col_origin).union(set(cov_list)))
        # col_drop.remove('rate')
        # col_drop = list(set(col_drop))
        df_re = df_here.drop(self.cov_list, axis = 1).dropna()
        
        rate_cols = [col for col in df_re.columns if col.startswith('rate_')]
        other_cols = [col for col in df_re.columns if not col.startswith('rate_')]
        col_sorted = other_cols + rate_cols
   
        return df_re[col_sorted]
    
    def _deal_cov_lag_test(self, df):
        """
        auto choose lag order by using correlation method, and then finish lag
        -----------------------------
        df : DataFrame
        cov_lagdict: lag dict
        """
        df_here = copy.deepcopy(df)
        col_origin = list(df_here.columns)
        ## 进行滞后处理
        for k in self.cov_lagdict.keys():
            if k not in df_here.columns:
                print(k," is not a column of data!!!")
            else:
                lag_num =  self.cov_lagdict[k]
                # print(k," lag ",lag_num," order")
                for l in range(1, lag_num+1):
                    df_here[f'{k}_{l}d'] = df_here[k].shift(l) 
        
        # col_drop = list(set(col_origin).union(set(self.cov_list)))
        # col_drop.remove('rate')
        # col_drop = list(set(col_drop))
        df_re = df_here.drop(self.cov_list, axis = 1).dropna()
        col_sorted = list(df_re.drop('rate', axis = 1).columns)
        col_sorted.append('rate')
        df_re = df_re[col_sorted]
        return df_re
    
    def _init_lag_func(self, max_rate_lag = 1, cov_list = None, max_cov_lag = 0):
        if max_cov_lag <= 0:
            raise Exception('the lag_mode is auto, but lag_mode and max_cov_lag is not matched!')
        self.max_rate_lag = max_rate_lag
        self.cov_list = cov_list
        self.max_cov_lag = max_cov_lag
        self.cov_lagdict = None

    def get_train_data(self, df, max_rate_lag = 1, cov_list = None, max_cov_lag = 0, pred_horizon = 1, 
                       validation = False, return_feature_names=True):
        """
        get the train and validation data from df dataframe
        ----------------------------------
        df : DataFrame of train data
        return_feature_names: 是否返回特征名称列表
        """
    # normalization
        self.pred_horizon = pred_horizon
        df_train = copy.deepcopy(df)
        self._init_scaler(df_train = df_train)
        df_train = self.maxmin_normalization(df_train)
    ## deal lag for covariates
        self._init_lag_func(max_rate_lag=max_rate_lag, cov_list=cov_list, max_cov_lag=max_cov_lag)
        df_train1 = self._deal_cov_lag_train(df_train)
        for i in range(pred_horizon):
            df_train1[f'rate_y{i}'] = df_train1['rate'].shift(-i)
        df_train1 = df_train1.drop('rate', axis = 1).dropna()
        x_train, y_train = df_train1.iloc[:,0:-pred_horizon], df_train1.iloc[:,-pred_horizon:]
    
    # Generate feature names if requested
        feature_names = list(x_train.columns)
    
        if validation is False:
            train_datadict = {'x_data': x_train,
                          'y_data': y_train}
            return train_datadict, self.max_lag_order, feature_names  # 总是返回feature_names
    
    # split train and validation data
        if validation is True:
            train_ind = np.random.choice(x_train.shape[0], size=int(x_train.shape[0]*0.85), replace=False)
            val_ind = np.setdiff1d(np.array(range(x_train.shape[0])), train_ind)
            x_train, y_train = x_train.iloc[train_ind,:], y_train.iloc[train_ind,:]
            x_val, y_val = x_train.iloc[val_ind,:], y_train.iloc[val_ind,:]
            train_datadict = {'x_data':x_train, 'y_data':y_train}
            val_datadict = {'x_data':x_val, 'y_data':y_val}
            if return_feature_names:
                return train_datadict, val_datadict, self.max_lag_order, feature_names
            else:
                return train_datadict, val_datadict, self.max_lag_order
        
        
    def get_test_data(self, df):
        # print(" --- for get_test_data function ---, origin shape = ", df.shape)
        df_t = self.maxmin_normalization(copy.deepcopy(df))
        # print(" ----- after normalization, shape = ", df_t.shape)
        df_t = self._deal_cov_lag_test(df_t)
        # print(" ----- after lag dealing, shape = ", df_t.shape)
        for i in range(self.pred_horizon):
            df_t[f'rate_y{i}'] = df_t['rate'].shift(-i)
        # print(" ----- after rate lag dealing, shape = ", df_t.shape)
        df_t = df_t.drop('rate', axis = 1).dropna()
        # print(" ----- after dropna, shape = ", df_t.shape)
        # print(df_t.shape)
        # x_test, y_test = df_t.iloc[:,0:-self.pred_horizon], df_t.iloc[:,-self.pred_horizon:]
        return df_t

class RFmodel():
    def __init__(self):
        self.best_param = {}
        self.model = None
        self.feature_names = None  # 确保初始化时存储特征名称

    def _init_model(self, random_state):
        self.model = RandomForestRegressor(random_state=random_state)
    
    def CV_train_(self, x_train, y_train, fold_num=5, param_dict={}, random_state=42, verbose=True):
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        
        model = RandomForestRegressor(random_state=random_state)
        model_cv = GridSearchCV(model, param_grid=param_dict, cv=fold_num, n_jobs=3)
        model_cv.fit(x_train, y_train)
        best_params = model_cv.best_params_
        self.model = RandomForestRegressor(**best_params)
        if verbose:
            print("For this model, the best parameters are ", best_params)

    def fit_(self, x_train, y_train, random_state):
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        elif isinstance(x_train, np.ndarray) and self.feature_names is None:
            # 如果没有特征名称，生成默认名称
            self.feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
            
        if self.model is None:
            self._init_model(random_state)
        else:
            self.best_param['random_state'] = random_state
            self.model = RandomForestRegressor(**self.best_param)
        
        # 确保传入的是数组值
        if hasattr(x_train, 'values'):
            x_train = x_train.values
        self.model.fit(x_train, y_train)

    def predict_(self, x_test):
        y_test_hat = self.model.predict(x_test)
        return y_test_hat
    
    def get_feature_importance(self):
        if self.model is None:
            raise Exception("Model not trained yet")
        
        # 获取特征重要性
        importance = self.model.feature_importances_
        
        # 获取特征名称
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            # 如果没有保存特征名称，生成默认名称
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return importance, feature_names
    def __init__(self):
        self.best_param = {}
        self.model = None
        self.feature_names = None

    def _init_model(self, random_state):
        self.model = RandomForestRegressor(random_state=random_state)
    
    def CV_train_(self, x_train, y_train, fold_num=5, param_dict={}, random_state=42, verbose=True):
        # 确保保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        
        model = RandomForestRegressor(random_state=random_state)
        model_cv = GridSearchCV(model, param_grid=param_dict, cv=fold_num, n_jobs=3)
        model_cv.fit(x_train, y_train)
        best_params = model_cv.best_params_
        self.model = RandomForestRegressor(**best_params)
        if verbose:
            print("For this model, the best parameters are ", best_params)

    def fit_(self, x_train, y_train, random_state):
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        elif isinstance(x_train, np.ndarray) and self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
            
        if self.model is None:
            self._init_model(random_state)
        else:
            self.best_param['random_state'] = random_state
            self.model = RandomForestRegressor(**self.best_param)
        
        # 确保传入的是数组值
        if hasattr(x_train, 'values'):
            x_train = x_train.values
        self.model.fit(x_train, y_train)

    def predict_(self, x_test):
        y_test_hat = self.model.predict(x_test)
        return y_test_hat
    
    def get_feature_importance(self):
        if self.model is None:
            raise Exception("Model not trained yet")
        
        # 获取特征重要性
        importance = self.model.feature_importances_
        
        # 获取特征名称
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return importance, feature_names
    def __init__(self):
        self.best_param = {}
        self.model = None
        self.feature_names = None  # Add this to store feature names

    def _init_model(self, random_state):
        self.model = RandomForestRegressor(random_state=random_state)
    

    def CV_train_(self, x_train, y_train, fold_num = 5, param_dict = {}, random_state = 42,verbose = True):
        """
        the tunning parameter list : [n_estimators, max_depth]
        ---------------------------------------
        """
        model = RandomForestRegressor(random_state = random_state)
        model_cv = GridSearchCV(model, param_grid=param_dict, cv=fold_num, n_jobs = 3)
        model_cv.fit(x_train, y_train)
        best_params = model_cv.best_params_
        self.model = RandomForestRegressor(**best_params) # best model
        if verbose == True:
            print("For this model, the best parameters are ", best_params)

    def fit_(self, x_train, y_train, random_state):
        if self.model is None:
            self._init_model(random_state)
        else:
            self.best_param['random_state'] = random_state
            self.model = RandomForestRegressor(**self.best_param) # best model
        self.model.fit(x_train,y_train)

    def predict_(self, x_test):
        y_test_hat = self.model.predict(x_test)
        return y_test_hat
    
    def output_model(self):
        return self.model
    
    def get_feature_importance(self):
        if self.model is None:
            raise Exception("Model not trained yet")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Get feature names
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            # Fallback if no feature names stored
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return importance, feature_names
    

class XGBmodel():
    def __init__(self) -> None:
        self.model = None
        self.best_param = {}

    def _init_model(self, random_state):
        self.model = XGBRegressor(random_state = random_state)   
        
    def CV_train_(self, x_train, y_train, fold_num = 5, param_dict = {}, iter_num = 80, verbose = True):
        """
        the tunning parameter list : [n_estimators, max_depth]
        ---------------------------------------
        """
        model = XGBRegressor()
        model_cv = RandomizedSearchCV(model, param_distributions=param_dict, n_iter=iter_num, cv=fold_num, n_jobs = 3)
        model_cv.fit(x_train, y_train)
        best_params = model_cv.best_params_
        self.best_param = best_params
        self.model = XGBRegressor(**best_params) # best model
        if verbose == True:
            print("For this model, the best parameters are ", best_params)

    def fit_(self, x_train, y_train, random_state):
        # x_data, y_data = copy.deepcopy(x_train), copy.deepcopy(y_train)
        if self.model is None:
            self._init_model(random_state)
        else:
            self.best_param['random_state'] = random_state
            self.model = XGBRegressor(**self.best_param)
        # train_ind = np.random.choice(x_data.shape[0], size=int(x_data.shape[0]*0.9), replace=False)
        # val_ind = np.setdiff1d(np.array(range(x_data.shape[0])), train_ind)
        # x_train, y_train = x_data[train_ind,:],y_data[train_ind,:]
        # x_val, y_val = x_data[val_ind,:], y_data[val_ind,:]
        # print("train_size = ", x_train.shape[0], ", val_size = ", x_val.shape[0])
        # self.model.fit(x_train,y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', early_stopping_rounds=5)
        self.model.fit(x_train, y_train) 

    def predict_(self, x_test):
        y_test_hat = self.model.predict(x_test)
        return y_test_hat
    
    def get_feature_importance(self):
        if self.model is None:
            raise Exception("Model not trained yet")
    
    # Get feature importance as a dictionary
        importance = self.model.get_booster().get_score(importance_type='weight')
    
    # Convert to array format sorted by feature names
        feature_names = self.model.get_booster().feature_names
        if feature_names is None:
            raise Exception("Feature names not available in the model")
        
        importance_array = np.array([importance.get(f, 0) for f in feature_names])
    
        return importance_array, feature_names  # Return both importance and names







class CatBoostModel():
    def __init__(self):
        self.models = []
        self.best_params = []
        self.n_outputs = 0
        self.feature_names = None  # 新增：存储特征名称

    def CV_train_(self, x_train, y_train, fold_num=5, param_dict={}, verbose=True):
        """
        CatBoost模型的交叉验证训练 - 支持多输出
        """
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        
        # 将输入转换为 numpy array
        if hasattr(x_train, 'values'):
            x_train_np = x_train.values
        else:
            x_train_np = x_train
            
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
        
        self.n_outputs = y_train_np.shape[1] if len(y_train_np.shape) > 1 else 1
        self.models = []
        self.best_params = []
        
        # 使用简单的参数，因为CatBoost的GridSearchCV较慢
        # 或者从param_dict中取第一个组合
        if param_dict:
            # 取每个参数的第一个值
            params = {}
            for k, v in param_dict.items():
                if isinstance(v, list) and len(v) > 0:
                    params[k] = v[0]
                else:
                    params[k] = v
        else:
            params = {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'verbose': False,
                'allow_writing_files': False
            }
        
        for i in range(self.n_outputs):
            if self.n_outputs > 1:
                y_train_i = y_train_np[:, i]
            else:
                y_train_i = y_train_np.ravel()
                
            # CatBoost支持多输出，但为了一致性，还是为每个输出单独训练
            model = cb.CatBoostRegressor(**params)
            model.fit(x_train_np, y_train_i)
            self.models.append(model)
            self.best_params.append(params)
            
        if verbose:
            print(f"Trained {self.n_outputs} CatBoost models with params: {params}")

    def fit_(self, x_train, y_train, random_state):
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        elif isinstance(x_train, np.ndarray) and self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
        
        # 将输入转换为 numpy array
        if hasattr(x_train, 'values'):
            x_train_np = x_train.values
        else:
            x_train_np = x_train
            
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
            
        self.n_outputs = y_train_np.shape[1] if len(y_train_np.shape) > 1 else 1
        self.models = []
        self.best_params = []
        
        params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'random_seed': random_state,
            'verbose': False,
            'allow_writing_files': False
        }
        
        for i in range(self.n_outputs):
            if self.n_outputs > 1:
                y_train_i = y_train_np[:, i]
            else:
                y_train_i = y_train_np.ravel()
                
            model = cb.CatBoostRegressor(**params)
            model.fit(x_train_np, y_train_i)
            self.models.append(model)
            self.best_params.append(params)

    def predict_(self, x_test):
        if not self.models:
            raise ValueError("Model not trained yet. Call fit_() first.")
        
        # 将输入转换为 numpy array
        if hasattr(x_test, 'values'):
            x_test_np = x_test.values
        else:
            x_test_np = x_test
        
        predictions = []
        for model in self.models:
            pred = model.predict(x_test_np)
            predictions.append(pred)
        
        # 将多个预测结果组合成 (n_samples, n_outputs) 的形状
        if self.n_outputs > 1:
            return np.column_stack(predictions)
        else:
            return predictions[0].reshape(-1, 1)
    
    def output_model(self):
        return self.models
    
    def get_feature_importance(self):
        """
        获取特征重要性
        返回特征重要性数组和特征名称
        """
        if not self.models:
            raise Exception("Model not trained yet")
        
        # 获取第一个模型的特征重要性（假设所有模型的特征重要性相似）
        model = self.models[0]
        
        # 获取特征重要性
        importance = model.get_feature_importance()
        
        # 获取特征名称
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            # 如果没有保存特征名称，生成默认名称
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return importance, feature_names




class LightGBMmodel():
    def __init__(self):
        self.models = []
        self.n_outputs = 0
        self.feature_names = None  # 新增：存储特征名称

    def CV_train_(self, x_train, y_train, fold_num=5, param_dict={}, verbose=True):
        """
        简化的LightGBM训练 - 不使用GridSearchCV
        """
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        
        # 将输入转换为 numpy array
        if hasattr(x_train, 'values'):
            x_train_np = x_train.values
        else:
            x_train_np = x_train
            
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
        
        self.n_outputs = y_train_np.shape[1] if len(y_train_np.shape) > 1 else 1
        self.models = []
        
        # 固定参数，或者从param_dict中取第一个组合
        if param_dict:
            # 取每个参数的第一个值
            params = {}
            for k, v in param_dict.items():
                if isinstance(v, list) and len(v) > 0:
                    params[k] = v[0]
                else:
                    params[k] = v
            params['verbose'] = -1
        else:
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'verbose': -1
            }
        
        for i in range(self.n_outputs):
            if self.n_outputs > 1:
                y_train_i = y_train_np[:, i]
            else:
                y_train_i = y_train_np.ravel()
                
            model = lgb.LGBMRegressor(**params)
            model.fit(x_train_np, y_train_i)
            self.models.append(model)
            
        if verbose:
            print(f"Trained {self.n_outputs} LightGBM models with params: {params}")

    def fit_(self, x_train, y_train, random_state):
        # 保存特征名称
        if hasattr(x_train, 'columns'):
            self.feature_names = list(x_train.columns)
        elif isinstance(x_train, np.ndarray) and self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(x_train.shape[1])]
        
        # 将输入转换为 numpy array
        if hasattr(x_train, 'values'):
            x_train_np = x_train.values
        else:
            x_train_np = x_train
            
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
            
        self.n_outputs = y_train_np.shape[1] if len(y_train_np.shape) > 1 else 1
        self.models = []
        
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': random_state,
            'verbose': -1
        }
        
        for i in range(self.n_outputs):
            if self.n_outputs > 1:
                y_train_i = y_train_np[:, i]
            else:
                y_train_i = y_train_np.ravel()
                
            model = lgb.LGBMRegressor(**params)
            model.fit(x_train_np, y_train_i)
            self.models.append(model)

    def predict_(self, x_test):
        if not self.models:
            raise ValueError("Model not trained yet. Call fit_() first.")
        
        # 将输入转换为 numpy array
        if hasattr(x_test, 'values'):
            x_test_np = x_test.values
        else:
            x_test_np = x_test
        
        predictions = []
        for model in self.models:
            pred = model.predict(x_test_np)
            predictions.append(pred)
        
        # 将多个预测结果组合成 (n_samples, n_outputs) 的形状
        if self.n_outputs > 1:
            return np.column_stack(predictions)
        else:
            return predictions[0].reshape(-1, 1)
    
    def output_model(self):
        return self.models
    
    def get_feature_importance(self):
        """
        获取特征重要性
        返回特征重要性数组和特征名称
        """
        if not self.models:
            raise Exception("Model not trained yet")
        
        # 获取第一个模型的特征重要性（假设所有模型的特征重要性相似）
        model = self.models[0]
        
        # 获取特征重要性
        importance = model.feature_importances_
        
        # 获取特征名称
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            # 如果没有保存特征名称，生成默认名称
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return importance, feature_names