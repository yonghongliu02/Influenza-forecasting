# 加载必要的库
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(readr)
library(purrr)
library(glmnet)

# 设置警告忽略
options(warn = -1)

# 定义AVG_ENSEMBLE类
AVG_ENSEMBLE <- function(model_list, forecasting_mode, origin_path, rolling_start_date) {
  
  self <- list()
  self$model_list <- model_list
  self$forecasting_mode <- forecasting_mode
 
  self$max_pred_horizon <- 8
 
  self$rolling_start_date <- rolling_start_date#滚动预测开始日期
  self$origin_path <- origin_path
  self$model_path <- paste0(self$origin_path, '/Results/Point/')
  
  # 1.获取原始日期:当前日期-提前预测的时间
  self$get_origin_date <- function(df) {
    df$date=as.Date(df$date)
    df$date - weeks(df$week_ahead)
  }
  
  
  # 2.获取最佳数据：过滤掉早于 rolling_start_date 的数据，并与真实值进行合并
  self$get_bst_data <- function(df_analysis, point_col) {
    df_analysis_bst <- df_analysis %>%
                   select(week_ahead, date, !!point_col, model)
    df_analysis_bst$date=as.Date(df_analysis_bst$date)
    df_analysis$date=as.Date(df_analysis$date)

    df_analysis_bst$date_origin <- self$get_origin_date(df_analysis_bst)
    
    df_analysis_bst <- df_analysis_bst %>% 
                    filter(date_origin >= self$rolling_start_date)
    
    df_analysis_bst$date <- as.Date(df_analysis_bst$date)
    
    df_analysis_bst <- df_analysis_bst %>% 
                   inner_join(df_analysis %>% 
                   filter(model == self$model_list[1])  %>% 
                   select(date, week_ahead, true), by = c("date", "week_ahead"))
    
    #write.csv(df_analysis_bst, paste0(origin_path,"/debug_my_data1.csv"), row.names = FALSE)
    
    return(df_analysis_bst)
  }
  
  # 3.计算RMSE
  self$compute_rmse <- function(df) {
    df1 <- df
    pred_horizons <- unique(df1$week_ahead)
    #对 pred_horizons迭代，计算预测不同步数的RMSE，最后平均
    rmse_list <- map_dbl(pred_horizons, ~ {
     
       df_t <- df1 %>% 
        filter(week_ahead == .x)
      
      rmse_t <- sqrt(mean((df_t$true - df_t$point_avg)^2))
      return(rmse_t)
    })
    return(mean(rmse_list))
  }
  
  # 4.计算加权RMSE：评估模型效果
  self$compute_weighted_rmse <- function(df, mode = 'Newton', lambda_ = lambda_) {
    df1 <- df
    if (mode == 'Newton') {
      df1 <- df1 %>% 
        group_by(week_ahead) %>% 
        mutate(rank = rank(desc(date)))#给每个预测时间排序
      
      df1$decay_coef <- exp(-lambda_ * df1$rank)#每个时间的权重系数
      df1$decay_coef[df1$decay_coef < 1e-3] <- 1e-3#最小值赋值
      df1$decay_weight <- df1$decay_coef / sum(df1$decay_coef)#权重
    }
    
    pred_horizons <- unique(df1$week_ahead)
    
    rmse_list <- map_dbl(pred_horizons, ~ {
      df_t <- df1 %>% 
        filter(week_ahead == .x)
      
      df_rmse <- ((df_t$true - df_t$point_avg)^2) * df_t$decay_weight
      rmse_t <- sqrt(mean(df_rmse))
      return(rmse_t)
    })
    return(mean(rmse_list))
  }
  
  # 5.SAE模型
  self$SAE <- function() {
    model_name <- 'SAE'
    
    #合并所有的模型结果
    df_test <- data.frame()
    for (m in self$model_list) {
      path_mt <- paste0(self$model_path, 'forecast_', m, '_', self$forecasting_mode, '.csv')
      df_mt_o <- read_csv(path_mt)
      df_mt_o$date <- as.Date(df_mt_o$date)
      df_mt_o <- df_mt_o %>% filter(date > self$rolling_start_date, date <= as.Date('2025-06-24'))
      df_mt_o$model <- m
      df_test <- bind_rows(df_test, df_mt_o)
    }
    
    df_full <- df_test
    df_full$date_origin <- self$get_origin_date(df_full)
    
    start_time <- Sys.time()
    
    date_analysises <- unique(df_full$date)
    date_analysises <- date_analysises[date_analysises >= as.Date('2023-10-06') ]
    
    #集成预测:按照分析的时间段循环
    re <- data.frame()
    for (l in seq_along(date_analysises)) {
      print(paste("-------------------------------------------  ", date_analysises[l], "  ----------------------------------------------------"))
      
      #提取截止到预测时间点的全部数据
      df_full_date_analysis <- df_full %>% 
        filter(date < date_analysises[l])
      
      #提取预测时间点当天的数据
      re_t <- df_full %>% 
        filter(date == date_analysises[l])
      
      print(paste(" ---- model_length =", length(unique(re_t$model))))
      
      if (length(unique(re_t$model)) == length(self$model_list)) {
        col_name <- 'point_avg'
        
        #合并真实值
        df_analysis_bst <- self$get_bst_data(df_full_date_analysis, col_name)
        
        #计算每个模型的RMSE：使用全部数据的
        res_rmse <- df_analysis_bst %>% 
          group_by(model) %>% 
          summarise(rmse = self$compute_rmse(cur_data()))
        
        #选择RMSE最小的两个模型
        rmse_top2 <- res_rmse %>% 
          arrange(rmse) %>% 
          head(2)
        
        ensemble_list <- rmse_top2$model
        
        print(ensemble_list)
        
        #简单平均
        re_t1 <- re_t %>% 
          filter(model %in% ensemble_list) %>% 
          group_by(date, week_ahead) %>% 
          summarise(across(everything(), mean))
        
        re <- bind_rows(re, re_t1)
      }
    }
    
    print(colnames(re))
    
    end_time <- Sys.time()
    print(paste("at the time<", format(start_time, '%Y-%m-%d %H:%M:%S'), ">, ", model_name, " begin,", " at the time<", format(end_time, '%Y-%m-%d %H:%M:%S'), "> finished."))
    print(paste("The running time totally =", as.numeric(difftime(end_time, start_time, units = "secs")), "seconds."))
    
    re1 <- re
    re1$var <- 'iHosp'
    re1$model <- model_name
    re1$region <- 'HK'
    re_final <- re1 %>% select(colnames(df_mt_o))
    re_final$date <- as.Date(re_final$date)
    
     write_csv(re_final, paste0(self$origin_path, '/Results/Point/forecast_', model_name, '_', self$forecasting_mode, '.csv'))
  }
  
  # 6. AWAE模型
  self$AWAE <- function(lambda_) {
    model_name <- 'AWAE'
    
    #加载所有模型的预测结果
    df_test <- data.frame()
    for (m in self$model_list) {
      path_mt <- paste0(self$model_path, 'forecast_', m, '_', self$forecasting_mode, '.csv')
      df_mt_o <- read_csv(path_mt)
      df_mt_o$date <- as.Date(df_mt_o$date)
      df_mt_o <- df_mt_o %>% filter(date > self$rolling_start_date, date <= as.Date('2025-06-24'))
      df_mt_o$model <- m
      df_test <- bind_rows(df_test, df_mt_o)
    }
    
    df_full <- df_test
    df_full$date_origin <- self$get_origin_date(df_full)
    
    start_time <- Sys.time()
    
    date_analysises <- unique(df_full$date)
    date_analysises <- date_analysises[date_analysises >= as.Date('2023-10-06') ]
    mu <- 0
    
    #遍历每个日期，进行集成预测
    re <- data.frame()
    for (l in seq_along(date_analysises)) {
      print(paste("-------------------------------------------  pred_date", date_analysises[l], "  ----------------------------------------------------"))
      df_full_date_analysis <- df_full %>% filter(date < date_analysises[l])
      re_t <- df_full %>% filter(date == date_analysises[l])
      print(paste(" ---- model_length =", length(unique(re_t$model))))
      re_t$coef_equal <- 0.0
      re_t$decay_coef <- 0.0
      
      for (wi in list(c(0), 1:self$max_pred_horizon)) {
        print(paste("-------------- week = ", wi, " ---------------"))
        if (length(unique(re_t %>% filter(week_ahead %in% wi) %>% pull(model))) == length(self$model_list)) {
          df_full_date_analysis_w <- df_full_date_analysis %>% filter(week_ahead %in% wi)
          df_test_t_wi <- re_t %>% filter(week_ahead %in% wi)
          
          col_name <- 'point_avg'
          df_analysis_bst <- self$get_bst_data(df_full_date_analysis_w, col_name)
          res_rmse <- df_analysis_bst %>% group_by(model) %>% 
            summarise(rmse = self$compute_weighted_rmse(cur_data(), mode = 'Newton', lambda_ = lambda_))
          
          rmse_top2 <- res_rmse %>% arrange(rmse) %>% head(2)
          ensemble_list <- rmse_top2$model
          
          print(ensemble_list)
          
          re_t1 <- re_t %>% filter(week_ahead %in% wi, model %in% ensemble_list) %>% group_by(date, week_ahead) %>% summarise(across(everything(), mean))
          re <- bind_rows(re, re_t1)
        } else {
          print("Not exist.")
          re_t <- re_t %>% filter(!week_ahead %in% wi)
        }
      }
    }
    
    print(colnames(re))
    
    end_time <- Sys.time()
    print(paste("at the time<", format(start_time, '%Y-%m-%d %H:%M:%S'), ">, ", model_name, " begin,", " at the time<", format(end_time, '%Y-%m-%d %H:%M:%S'), "> finished."))
    print(paste("The running time totally =", as.numeric(difftime(end_time, start_time, units = "secs")), "seconds."))
    
    re1 <- re
    re1$var <- 'iHosp'
    re1$model <- model_name
    re1$region <- 'HK'
    re_final <- re1 %>% select(colnames(df_mt_o))
    re_final$date <- as.Date(re_final$date)
    
     write_csv(re_final, paste0(self$origin_path, '/Results/Point/forecast_', model_name, '_', self$forecasting_mode, '.csv'))
  }
  
  return(self)
}

# 主程序
model_list <- c(
  'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'LGBM_rolling',
  'CB_rolling',

'LSTM_direct_multioutput_rolling',
'GRU_direct_multioutput_weighted_rolling'
)
mode <- 'test8_2023'


origin_path <- dirname(dirname(getwd()))

rolling_start_date <- as.Date('2023-07-06')
decay_mode <- 6
lambda_ <- -log(0.01) / decay_mode

# 创建AVG_ENSEMBLE实例
my_avg_ensemble <- AVG_ENSEMBLE(model_list = model_list,
                                forecasting_mode = mode,
                                origin_path = origin_path,
                                rolling_start_date = rolling_start_date)

# 运行SAE和AWAE模型
my_avg_ensemble$SAE()
my_avg_ensemble$AWAE(lambda_ = lambda_)
