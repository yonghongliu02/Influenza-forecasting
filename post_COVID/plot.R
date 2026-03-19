library(plyr)
library(dplyr)
library(tidyverse)
library(ISOweek)
library(locpol)
library(forecast)
library(scoringutils)
library(tseries)
library(cowplot)
library(tidyr)
library(rugarch)
library(plyr)
library(dplyr)
library(tidyverse)
library(ISOweek)
library(locpol)
library(forecast)
library(scoringutils)
library(tseries)
library(cowplot)
library(tidyr)
library(rugarch)
library(tsibble)
#--------------------------1.individule----------------------------

model_list <- c(
  "Baseline",
  'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'LGBM_rolling',
  'CB_rolling',
  "LSTM_direct_multioutput_rolling",
  'GRU_direct_multioutput_weighted_rolling'
)

path=setwd("../../Results/Point")

df_test <- data.frame()

for (m in model_list) {
  path_mt <- paste0(path, '/forecast_', m, '_', "test8_2023", '.csv')
  df_mt_o <- read_csv(path_mt)
  df_mt_o$date <- as.Date(df_mt_o$date)
  df_mt_o <- df_mt_o[which(df_mt_o$date>=as.Date(yearweek("2023 W27"))),]
  df_mt_o$model <- m
  print(m)
  print(min(df_mt_o$date))
  print(max(df_mt_o$date))
  df_test <- bind_rows(df_test, df_mt_o)
}

table(df_test$model)


df_test=df_test[which(df_test$week_ahead==0|
                        df_test$week_ahead==4|
                        df_test$week_ahead==8),]


df_test$model=factor(df_test$model,levels=model_list)
ggplot(df_test, aes(x = date)) +
  geom_line(aes(y = true, fill="black"), linewidth = 0.5) +
  geom_line(aes(y = point, color = factor(week_ahead)), linewidth = 0.5) +
  facet_wrap(~ model, scales = "free_y",nrow=5) +
 # scale_color_viridis_d(name = "Ahead") +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))







#--------------------------2.ensemble-----------------------------
model_list <- c(
  "SAE",
  'AWAE'


)

#path=setwd("../../Results/Point")

df_test <- data.frame()

for (m in model_list) {
  path_mt <- paste0(path, '/forecast_', m, '_', "test8_2023", '.csv')
  df_mt_o <- read_csv(path_mt)
  df_mt_o$date <- as.Date(df_mt_o$date)
  df_mt_o <- df_mt_o[which(df_mt_o$date>=as.Date(yearweek("2024 W27"))),]
  df_mt_o$model <- m
  print(m)
  print(min(df_mt_o$date))
  print(max(df_mt_o$date))
  df_test <- bind_rows(df_test, df_mt_o)
}

table(df_test$model)


df_test=df_test[which(df_test$week_ahead==0|
                        df_test$week_ahead==4|
                        df_test$week_ahead==8),]


df_test$model=factor(df_test$model,levels=model_list)
ggplot(df_test, aes(x = date)) +
  geom_line(aes(y = true, fill="black"), linewidth = 0.5) +
  geom_line(aes(y = point, color = factor(week_ahead)), linewidth = 0.5) +
  facet_wrap(~ model, scales = "free_y",nrow=5) +
  # scale_color_viridis_d(name = "Ahead") +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))


