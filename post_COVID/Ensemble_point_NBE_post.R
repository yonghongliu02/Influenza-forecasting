# 加载必要的库
library(dplyr)
library(tidyr)
library(lubridate)
library(glmnet)
library(purrr)
library(readr)

# 设置警告忽略
options(warn = -1)

# 定义BLENDING类
BLENDING <- function(model_list, forecasting_mode, origin_path, rolling_start_date) {
  
  self <- list()
  self$model_list <- model_list
  self$forecasting_mode <- forecasting_mode

  self$max_pred_horizon <- 8
  
  self$rolling_start_date <- as.Date(rolling_start_date)
  self$origin_path <- origin_path
  self$model_path <-  paste0(self$origin_path, '/Results/Point')
  
  # 1. 计算原始日期
  self$get_origin_date <- function(df) {
    df$date=as.Date(df$date)
    df$date - weeks(df$week_ahead)
  }
  
  # 2. 获取基准数据
  self$get_bst_data <- function(df_analysis, model_list, point_col) {
    
    df_analysis$date=as.Date(df_analysis$date)
    
    #合并
    df_analysis_bst <- df_analysis %>%
      select(week_ahead, date, !!sym(point_col), model) %>%
      mutate(date_origin = self$get_origin_date(.)) %>%
      filter(date_origin >= self$rolling_start_date) %>%
      pivot_wider(names_from = model, values_from = !!sym(point_col)) %>%
      mutate(date = as.Date(date))
    
    true_values <- df_analysis %>%
      filter(model == model_list[1]) %>%
      select(date, week_ahead, true)
    
    df_analysis_bst <- df_analysis_bst %>%
      inner_join(true_values, by = c("date", "week_ahead")) %>%
      drop_na()
    

    return(df_analysis_bst)
  }
  
  # 3. 获取NBE模型系数
  self$get_bst_coef_nbe <- function(df_bst_analysis, model_list) {
    set.seed(2023)
    alpha_range <- seq(1, 29) / 10
    
    # 普通Lasso回归
    df_bst_p1 <- df_bst_analysis %>% select(all_of(model_list), true)%>%
      mutate(across(everything(), as.numeric))  # 强制转换所有列为数值型
    
    x <- as.matrix(df_bst_p1[model_list])
    y <- df_bst_p1$true
    
    #交叉验证选择最佳lambda
    lasso_cv <- cv.glmnet(x, y, alpha = 1, lambda = alpha_range, 
                          intercept = FALSE, nfolds = 5)
    best_alpha <- lasso_cv$lambda.min
    cat("best_alpha = ", best_alpha, "\n")
    
    # 使用最佳lambda拟合模型
    lasso1 <- glmnet(x, y, alpha = 1, lambda = best_alpha, intercept = FALSE)
    coef1 <- as.vector(coef(lasso1))
    
    for (i in seq_along(model_list)) {
      cat("-- ", model_list[i], " : ", coef1[i], "\n")
    }
    cat("---- val score = ", 
        lasso_cv$glmnet.fit$dev.ratio[which(lasso_cv$glmnet.fit$lambda == best_alpha)], 
        "\n")
    
   
    
    return(list(lasso1 = lasso1, best_alpha1 = best_alpha))
  }
  
  # 4. NBE模型
  self$NBE <- function() {
    model_name <- 'NBE'
    
    # 1.加载所有模型数据
    df_test <- map_dfr(self$model_list, ~ {
      path_mt <- file.path(self$model_path, paste0('forecast_', .x, '_', self$forecasting_mode, '.csv'))
      read_csv(path_mt) %>%
        mutate(date = as.Date(date),
               model = .x) %>%
        filter(date > self$rolling_start_date, 
               date <= as.Date('2025-06-24'))
    })
    
    df_full <- df_test
    start_time <- Sys.time()
    
    #2. 提取所有的滚动日期
    date_analysises <- unique(df_full$date)
    date_analysises <- date_analysises[date_analysises >= as.Date('2023-10-06') ]
    

    
   # date_analysises=date_analysises[!date_analysises %in% c(as.Date("2017-07-03"))]
    
    mu <- 1
    re <- data.frame()
    
    #3. 滚动每一个预测的日期
    for (current_date in date_analysises) {
      cat("-------------------------------------------  pred_date",
          format(as.Date(current_date, origin = "1970-01-01"), "%Y-%m-%d"),
          "  ----------------------------------------------------\n")
      
      #3.1筛选历史数据<current_date
      df_full_date_analysis <- df_full %>% 
        filter(date < current_date)
      
      print(current_date)

      
      #3.2筛选历史数据=current_date
      df_test_t <- df_full %>% 
        filter(date == current_date)
      
      #3.3整理结果表，新增加预测值的列
      re_t <- df_test_t %>%
        group_by(date, week_ahead) %>%
        summarise(true = mean(true), .groups = 'drop') %>%
        mutate(pred_equal = 0.0, pred_decay = 0.0)
      

      
      #3.4 滚动预测每一个ahead week
      for (wi in 0:self$max_pred_horizon) {
        
        cat("-------------- week = ", wi, " ---------------\n")
        if (length(unique(df_test_t$model[df_test_t$week_ahead %in% wi])) == length(self$model_list)) {
          
          
          #3.4.1 筛选指定预测步长的历史数据
          df_full_date_analysis_w <- df_full_date_analysis %>% 
            filter(week_ahead %in% wi)
          
          #3.4.2 当天第week的数据
          df_test_t_wi <- df_test_t %>% 
            filter(week_ahead %in% wi)
          col_name <- 'point_avg'
          
          #3.4.3 处理转换基准数据
          df_analysis_bst <- self$get_bst_data(df_full_date_analysis_w, self$model_list, col_name)
          
          #3.4.4 建模
          lasso_results <- self$get_bst_coef_nbe(df_analysis_bst, self$model_list)
          
          #3.4.5 处理当天转换基准数据
          df_test_t_wi_bst <- self$get_bst_data(df_test_t_wi, self$model_list, col_name)
          
          x_new <- as.matrix(df_test_t_wi_bst[self$model_list])
          
          model_benchmark <- df_test_t_wi %>%
            group_by(model) %>%
            summarise(count = n(), .groups = 'drop') %>%
            filter(count == min(count)) %>%
            pull(model) %>%
            first()
          
          wi1 <- df_test_t_wi %>% 
            filter(model == model_benchmark) %>%
            pull(week_ahead) %>%
            unique()
          
          re_t1 <- re_t %>% filter(week_ahead %in% wi1)
          
          re_t1$pred_equal <- predict(lasso_results$lasso1, newx = x_new, s = lasso_results$best_alpha1)
          
          re_t1[[col_name]] <- mu * re_t1$pred_equal 
          re <- bind_rows(re, re_t1)
        } else {
          cat("Not exist.\n")
          re_t <- re_t %>% filter(!week_ahead %in% wi)
        }
      }
    }
    
    end_time <- Sys.time()
    cat("Runtime:", as.numeric(difftime(end_time, start_time, units = "secs")), "seconds\n")
    

    re1 <- re %>%
      mutate(var = 'iHosp',
             model = model_name,
             region = 'HK',
             point = point_avg,
             date = as.Date(date),
             point_avg = pmax(0, point_avg),
             point = pmax(0, point)) %>%
      select(any_of(names(df_test))) 
    
    write_csv(re1, paste0(self$origin_path, '/Results/Point/forecast_', model_name, '_', self$forecasting_mode, '.csv'))
    
    return(re1)
  }
  
  # 5. 获取AWBE模型系数
  self$get_bst_coef_awbe <- function(df_bst_analysis, model_list, lambda_) {
    set.seed(2023)
    alpha_range <- seq(1, 29) / 10
    
   
    # 加权Lasso回归
    df_bst_p2 <- df_bst_analysis %>% 
      select(all_of(model_list), true, week_ahead, date) %>%
      group_by(week_ahead) %>%
      mutate(rank = rank(desc(date))) %>%
      ungroup() %>%
      mutate(decay_coef = exp(-lambda_ * rank),
             decay_coef = ifelse(decay_coef < 1e-3, 1e-3, decay_coef))
    
    x <- as.matrix(df_bst_p2[model_list])
    y <- df_bst_p2$true
    weights <- df_bst_p2$decay_coef
    
    lasso_cv_weighted <- cv.glmnet(x, y, alpha = 1, lambda = alpha_range,
                                   intercept = FALSE, nfolds = 5, weights = weights)
    best_alpha_weighted <- lasso_cv_weighted$lambda.min
    cat("加权Lasso - best_alpha =", best_alpha_weighted, "\n")
    
    lasso_model_weighted <- glmnet(x, y, alpha = 1, lambda = best_alpha_weighted, 
                                   intercept = FALSE, weights = weights)
    coef2 <- as.vector(coef(lasso_model_weighted))[-1]
    
    # 打印加权Lasso的系数
    cat("加权Lasso系数:\n")
    for (i in seq_along(model_list)) {
      cat("--", model_list[i], ":", coef2[i], "\n")
    }
    cat("验证分数(R²):", 
        lasso_cv_weighted$glmnet.fit$dev.ratio[which(lasso_cv_weighted$lambda == best_alpha_weighted)], 
        "\n\n")
    
    return(list( coef2 = coef2))
  }
  
  # 6. AWBE模型
  self$AWBE <- function(lambda_) {
    model_name <- 'AWBE'
    
    # 加载所有模型数据
    df_test <- map_dfr(self$model_list, ~ {
      path_mt <- file.path(self$model_path, paste0('forecast_', .x, '_', self$forecasting_mode, '.csv'))
      read_csv(path_mt) %>% 
        mutate(date = as.Date(date),
               model = .x) %>% 
        filter(date > self$rolling_start_date,
               date <=as.Date('2025-06-24')) %>% 
        select(date, true, week_ahead, point, point_avg, model)
    })
    
    df_full <- df_test
    start_time <- Sys.time()
    
    date_analysises <- unique(df_full$date)
    date_analysises <- date_analysises[date_analysises >= as.Date('2023-10-06') ]
    mu <- 0
    
    re <- data.frame()
    for (current_date in date_analysises) {
      cat("-------------------------------------------  pred_date", 
          format(as.Date(current_date, origin = "1970-01-01"), "%Y-%m-%d"),
          "  ----------------------------------------------------\n")
      
      df_full_date_analysis <- df_full %>% 
        filter(date < current_date)
      
      df_test_t <- df_full %>% 
        filter(date == current_date)
      
      re_t <- df_test_t
      
      for (wi in 0:self$max_pred_horizon) {
        cat("-------------- week = ", wi, " ---------------\n")
        if (length(unique(df_test_t$model[df_test_t$week_ahead %in% wi])) == length(self$model_list)) {
          df_full_date_analysis_w <- df_full_date_analysis %>% 
            filter(week_ahead %in% wi)
          col_name <- 'point_avg'
          
          df_analysis_bst <- self$get_bst_data(df_full_date_analysis_w, self$model_list, col_name)
          coef_results <- self$get_bst_coef_awbe(df_analysis_bst, self$model_list, lambda_)
          
          # 归一化系数
          coef_norm <- coef_results$coef2 / sum(coef_results$coef2)
          
          # 对每个模型应用归一化后的系数
          df_test_t_wi <- df_test_t %>% 
            filter(week_ahead %in% wi) %>% 
            mutate(weight = case_when(
              model == model_list[1] ~ coef_norm[1],
              model == model_list[2] ~ coef_norm[2],
              model == model_list[3] ~ coef_norm[3],
              model == model_list[4] ~ coef_norm[4],
              model == model_list[5] ~ coef_norm[5],
              model == model_list[6] ~ coef_norm[6],
              model == model_list[7] ~ coef_norm[7],
              model == model_list[8] ~ coef_norm[8]
            )) %>% 
            mutate(weighted_point = point_avg * weight,
                   weighted_point_raw = point * weight)
          
          # 汇总结果
          re_t_agg <- df_test_t_wi %>% 
            group_by(date, week_ahead) %>% 
            summarise(
              point_avg = sum(weighted_point),
              point = sum(weighted_point_raw),
              true = first(true),
              .groups = 'drop'
            )
          
          write_csv( re_t_agg,"re_t_agg.csv")
          
          re <- bind_rows(re, re_t_agg)
      
        } else {
          cat("Not exist.\n")
        }
      }
    }
    
    end_time <- Sys.time()
    cat("Runtime:", as.numeric(difftime(end_time, start_time, units = "secs")), "seconds\n")
    
    # 最终处理
    re_final <- re %>% 
      mutate(
        var = 'iHosp',
        model = model_name,
        region = 'HK',
        date = as.Date(date),
        point_avg = pmax(0, point_avg),
        point = pmax(0, point)
      ) %>% 
      # 修改为安全的选择方式
      select(any_of(names(df_test))) %>%
      arrange(date, week_ahead)
    
    write_csv(re_final, paste0(self$origin_path, '/Results/Point/forecast_', model_name, '_', self$forecasting_mode, '.csv'))
    
    return(re_final)
  }
  
  
  return(self)
}

# 主程序
model_list <- c(
  'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'lightgbm_rolling',
  'catboost_rolling',
  #'LSTM_direct_multioutput_rolling',
  #'GRU_direct_multioutput_weighted_rolling',
  'LSTM_direct_multioutput_rolling',
  'GRU_direct_multioutput_weighted_rolling'
)
mode <- 'test8_2023'
origin_path <-dirname(dirname(getwd()))

rolling_start_date <- '2023-07-06'
decay_mode <- 12
lambda_ <- -log(0.01) / decay_mode


# 创建BLENDING实例
my_blending <- BLENDING(model_list = model_list,
                        forecasting_mode = mode,
                        origin_path = origin_path,
                        rolling_start_date = rolling_start_date)

# 运行模型
nbe_results <- my_blending$NBE()
awbe_results <- my_blending$AWBE(lambda_ = lambda_)
