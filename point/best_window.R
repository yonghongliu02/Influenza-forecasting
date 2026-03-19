# 生成 total_interval_window.csv 的 R 脚本

library(dplyr)
library(tidyverse)
library(tsibble)

# 设置参数
std_mode <- 'ydiff'  # 与 interval_result_generate.R 中一致
mode <- 'test8'      # 与你的文件一致

# 模型列表 - 与 interval_result_generate.R 一致
models <- c(
  'baseline',
  'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'LGBM_rolling',
  'CB_rolling',
  'LSTM_direct_multioutput_rolling',
  'GRU_direct_multioutput_weighted_rolling',
  'SAE',
  'AWAE'
)

# 简化的模型名称 - 用于输出
model_name_list <- sapply(models, function(st) {
  strsplit(st, '_', fixed = TRUE)[[1]][1]
})

# 窗口长度范围 - 与 std_generate.py 一致
window_intervals <- c(5, seq(8, 50, 2))

# 初始化结果数据框
results <- tibble()

# 遍历所有模型、窗口长度和预测步长
for (model in models) {
  for (window_size in window_intervals) {
    # 构建文件路径
    path_ <- paste0('../../Results/Interval_', std_mode, '_raw/interval', 
                    window_size, '_', model, '_', mode, '.csv')
    
    # 如果文件存在，则读取并处理
    if (file.exists(path_)) {
      df <- read.csv(path_, stringsAsFactors = FALSE) %>%
        mutate(date = as.Date(date))
      ##删除窗口
      df=df[-which(df$date  <=as.Date(yearweek("2017 W27"))| df$date>=as.Date(yearweek("2020 W26"))),]
      
      # 计算每个 week_ahead 的覆盖率和WIS
      for (week_ahead in unique(df$week_ahead)) {
        df_week <- df %>% filter(week_ahead == !!week_ahead)
        
        # 计算90%区间的覆盖率 (使用 lower_5 和 upper_95)
        coverage_90 <- mean(df_week$true >= df_week$lower_10 & 
                              df_week$true <= df_week$upper_10, na.rm = TRUE)
        
        # 计算50%区间的覆盖率 (使用 lower_25 和 upper_75)
        coverage_50 <- mean(df_week$true >= df_week$lower_50 & 
                              df_week$true <= df_week$upper_50, na.rm = TRUE)
        
        # 计算Winkler Score (90%区间)
        winkler_score <- mean(
          ifelse(df_week$true >= df_week$lower_10 & df_week$true <= df_week$upper_10,
                 df_week$upper_10 - df_week$lower_10,
                 df_week$upper_10 - df_week$lower_10 + 
                   2/0.1 * ifelse(df_week$true < df_week$lower_10, 
                                  df_week$lower_10 - df_week$true,
                                  df_week$true - df_week$upper_10))
        )
        
        # 添加到结果
        results <- results %>%
          bind_rows(tibble(
            model = model_name_list[model],
            week_ahead = week_ahead,
            window_interval = window_size,
            avg_wis = winkler_score,
            cov50 = coverage_50,
            cov90 = coverage_90
          ))
      }
    }
  }
}

# 选择最优窗口长度 - 选择cov90最接近90%的窗口
optimal_windows <- results %>%
  group_by(model, week_ahead) %>%
  mutate(coverage_diff = abs(cov90 - 0.9)) %>%
  arrange(coverage_diff) %>%
  slice(1) %>%
  ungroup() %>%
  select(model, week_ahead, choose_window_interval = window_interval, avg_wis, cov50, cov90)

# 保存结果
write.csv(optimal_windows, file = '../../Results/Interval_ydiff_raw/total_interval_window.csv', row.names = FALSE)

# 输出结果预览
print(head(optimal_windows))
