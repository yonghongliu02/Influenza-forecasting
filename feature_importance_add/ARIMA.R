########################################
# 数据准备
########################################
# 数据准备
library(plyr)
library(dplyr)
library(tidyverse)
library(ISOweek)
library(locpol)
library(forecast)
library(scoringutils)
library(tseries)
library(cowplot)
library(tsibble)

data_rt<-readRDS("../../Data/model_long_data1022.rds")


data_rt$region="Beijing"
data_rt$date=as.Date(data_rt$week)
data_rt$date_analysis=as.Date(data_rt$date_analysis)


data_rt=data_rt[which(data_rt$date_analysis>=as.Date(yearweek("2023 W27"))),]


data_rt$iHosp<- data_rt$ILI_proxy_10log
data_rt$iHosp_smooth <- data_rt$ILI_proxy_10log

regions="Beijing"
max_lag = 14
model_name = 'ARIMA_rolling'
covar_all<-c("mean_temperature", "rh","absenteeism")
remove_last_n <- 1
# maximum prediction horizon
max_prediction_horizon <- 8+remove_last_n
mode = 'test8_2023_add'

dates_analysis = unique(data_rt$date_analysis)


# 建模
library(tibble)
res<-tibble()

for (iDate in seq_along(dates_analysis)){
  # iDate = 1
  set.seed(iDate)
  print(dates_analysis[iDate])
  dat_train<-data_rt %>%
    filter(date_analysis==dates_analysis[iDate]) %>%
    mutate(iHosp_r_covar=iHosp)
  
  # Estimate best lags for covariates
  res_lag<-tibble()
  list_col<-sort(covar_all)
  n_predictors <- length(list_col)
  
  if (n_predictors>0){
    for (iCol in seq_along(list_col)){
      # iCol = 1
      mycol<-list_col[iCol]
      
      for (lag in 1:max_lag){
        # lag = 1
        sub1<-dat_train
        sub2<-sub1
        sub2$date<-sub2$date+lag*7
        
        sub<-left_join(sub1[,c("date","region","iHosp")],sub2[,c("date","region",mycol)], by = c("date", "region"))
        sub<-sub %>% drop_na(iHosp,mycol) %>%
          as.data.frame()
        
        cor<-cor.test(y=sub$iHosp,x=sub[,mycol],method="pearson")$estimate
        
        res_lag_tmp<-data.frame(pearson=cor,region="regions",lag=lag,var=mycol)
        res_lag<-bind_rows(res_lag,res_lag_tmp)
        
      }
    }
    
    
    best_lag<-res_lag %>% 
      group_by(var) %>%
      dplyr::slice(which.max(abs(pearson)))
    if ("iHosp_r_covar" %in% list_col){
      best_lag$lag[best_lag$var=="iHosp_r_covar"]<-1
    }
    
    list_lag<-best_lag$lag
    print(list_lag)
  }
  
  for (iRegion in seq_along(regions)) {
    # iRegion = 1
    myreg<-regions[iRegion]
    # print(myreg) 
    
    dat_train_reg<-dat_train %>%
      filter(region==myreg)
    
    # Prepare lagged data
    sub<-dat_train_reg[,c("date","region","iHosp","weekid",'monthid')]
    sub1<-dat_train_reg[,c("date","region","iHosp","weekid",'monthid',list_col)]
    lag_list_col = c()
    if (n_predictors>0){
      for (i_col in 1:length(list_lag)){
        col_name = best_lag$var[i_col]
        best_lag_val = best_lag$lag[i_col]
        print(paste0(col_name,' best_lag_val - ', best_lag_val))
       
         i_lag = best_lag_val
          sub2<-sub1
          sub2$date<-sub2$date+i_lag*7
          
          sub<-left_join(sub,sub2[,c("date","region",col_name)],by=c("region"="region","date"="date"))
          colnames(sub)[ncol(sub)] = paste0(col_name,i_lag)
          lag_list_col = c(lag_list_col, paste0(col_name,i_lag))
        
        
      }
    }
    sub<-sub %>% drop_na(iHosp,all_of(lag_list_col)) %>%
      dplyr::select(-date,-region)
    
    # Fit model 
    mod<-auto.arima(sub[,"iHosp"], xreg=as.matrix(sub[,c(2:ncol(sub))]),method="ML",allowdrift = F) # 
    
    re.coef = as.data.frame(mod$coef)
    coef.name <- rownames(re.coef)
    re.coef <- cbind(coef.name,re.coef)
    colnames(re.coef) = c("feature_name", "feature_coef")
    re.coef$date = dates_analysis[iDate]
    
    re.coef <- re.coef %>%
      mutate(
        predictor = str_extract(feature_name, "[:alpha:]+" ),
        lag_order = str_extract(feature_name, "[0-9]+$"),
        lag_order = ifelse(is.na(lag_order), NA, paste0(lag_order, "d"))
      )
    
    res<-rbind(res,re.coef)
    
  }
  
}


###标准化
res <- res %>%
  mutate(
    lag_order = if_else(predictor %in% c("monthid", "weekid"), "1d", lag_order)
  )
# 1. 过滤掉非外部变量（AR, MA, weekid, monthid 等）
res_filtered <- res %>%
  filter(!predictor %in% c("intercept","ma")) 

# 2. 每个模型内标准化协变量的系数（不管日期，只管“模型一次”的单位）
#    假设每个 date 就是一次模型
res_standardized <- res_filtered %>%
  group_by(date) %>%
  mutate(
    abs_coef = abs(feature_coef),
    norm_coef = abs_coef / sum(abs_coef, na.rm = TRUE)
  ) %>%
  ungroup()

# 3. 计算每个变量-滞后组合在所有模型中的平均解释性
res_summary <- res_standardized %>%
  group_by(predictor, lag_order) %>%
  summarise(
    mean_importance = mean(norm_coef, na.rm = TRUE),
    sd_importance = sd(norm_coef, na.rm = TRUE),
    times_included = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_importance))

#plot
res_summary %>%
  mutate(lag_numeric = as.numeric(gsub("d", "", lag_order))) %>%
  ggplot(aes(x = lag_numeric, y = predictor, fill = mean_importance)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Average Feature Importance by Lag",
       x = "Lag (days)", y = "Predictor")


## save
path = paste0('../../Results/FI_add/fi_', model_name,'_',mode,'.csv')
write.csv(res, path, row.names = FALSE)

path = paste0('../../Results/FI_add/fi_', model_name,'_',mode,'_final.csv')

write.csv(res_summary, path, row.names = FALSE)





