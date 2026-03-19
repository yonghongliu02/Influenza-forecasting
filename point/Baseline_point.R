####################
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
library(tsibble)


data_rt<-readRDS("../../Data/model_long_data1022.rds")
data_rt$region="Beijing"

data_rt$date=as.Date(data_rt$week)
data_rt$date_analysis=as.Date(data_rt$date_analysis)

#不扣除cov
#data_rt=data_rt[-which(data_rt$date_analysis>=as.Date(yearweek("2020 W27"))&
#                         data_rt$date_analysis<=as.Date(yearweek("2023 W26") )),] 
data_rt=data_rt[-which(data_rt$date_analysis>=as.Date(yearweek("2024 W27"))),]




data_rt$iHosp_smooth <- data_rt$ILI_proxy_10log

mode = 'test8'
dates_analysis = unique(data_rt$date_analysis)



remove_last_n <- 1
# maximum prediction horizon
max_prediction_horizon<-8+remove_last_n
date_max<-max(dates_analysis)+max_prediction_horizon-remove_last_n
model_name = 'Baseline'

library(msm)

# function taken from https://github.com/sbfnk/covid19.forecasts.uk 
null_model_forecast_quantiles <- function(values, horizon, truncation = 0)
{
  # values = dat_train$iHosp_smooth
  # horizon = max_prediction_horizon
  # truncation = 0
  if (truncation > 0) {
    values <- head(values, -truncation)
  }
  
  tnorm_unc_fit <- function(x, mean, true) {
    return(-sum(dtnorm(x = true, mean = mean, sd = x, lower = 0, log = TRUE)))
  }
  
  quantiles <- c(0.01, 0.025, seq(0.05, 0.95, by = 0.05), 0.975, 0.99)
  ts <- tail(values, min(horizon, length(values)))
  
  if (length(ts) > 1) {
    ## null model
    null_ts <- rep(last(ts), horizon + truncation)
    
    interval <- c(0, max(abs(diff(ts))))
    
    if (diff(interval) > 0 ) {
      tnorm_sigma <-
        optimise(tnorm_unc_fit, interval = interval,
                 mean = head(ts, -1),
                 true = tail(ts, -1))
      
      quant <- qtnorm(p = quantiles, mean = last(ts),
                      sd = tnorm_sigma$minimum, lower = 0)
    } else {
      quant <- rep(last(ts), length(quantiles))
    }
  } else {
    quant <- rep(last(ts), length(quantiles))
  }
  quant = exp(quant)/10  #return the true inverse-log rate
  names(quant) <- quantiles
  
  return(quant)
}


library(tibble)
res <- tibble()
library(dplyr)
iRegion = 'Beijing'

for (iDate in seq_along(dates_analysis)) {
   #iDate = 1
  
  #筛选训练集 
  dat_train<-data_rt %>%
    filter(date_analysis==dates_analysis[iDate], 
           region==iRegion)  %>%
    as.data.frame()
  
  #预测
  quant <-
    null_model_forecast_quantiles(dat_train$iHosp_smooth, horizon = max_prediction_horizon,
                                  truncation = 0)
  #命名
  names(quant)<-c(paste0("lower_",c(2,5,seq(10,90,by=10))),"point",paste0("upper_",rev(c(2,5,seq(10,90,by=10)))))
  
  
  res_tmp <- data.frame(
    var = "iHosp",
    region = iRegion, 
    date = dates_analysis[iDate],
    prediction_horizon = seq(1,max_prediction_horizon)-remove_last_n,
    t(quant))
  
  res <- rbind(res, res_tmp)
  
  
}

model = 'baseline'
res1 <- res %>%
  mutate(date=as.Date(date)) %>%
  mutate(date_pred=date+prediction_horizon*7) %>%
  mutate(model=model) 

#预测值
res1 = res1 %>%
  filter(date_pred>=min(dates_analysis))  %>%
  filter(date_pred<=max(dates_analysis))

#观测值
dat_true<-data_rt %>%
  filter(date_analysis==max(data_rt$date_analysis)) %>%
  filter(date>=min(dates_analysis)-remove_last_n & date<=date_max) %>%
  dplyr::select(date,region,iHosp_smooth) %>%
  pivot_longer(cols=c("iHosp_smooth"),
               names_to="var",values_to="smooth_value") %>%
  mutate(var=gsub("_smooth","",var)) %>%
  filter(!is.element(region,c("COR","national")))

#逆转换
dat_true$smooth_value = exp(dat_true$smooth_value)/10

#合并2个数据集
res1<-res1 %>%
  left_join(dat_true,by=c("date_pred"="date","region"="region","var"="var")) 

res1 = res1[ , -which(colnames(res1) %in% c("date"))]

#
res1 = res1 %>% 
  rename(date = date_pred) %>% 
  rename(true = smooth_value) %>% 
  rename(week_ahead = prediction_horizon) %>%
  mutate(point_avg = point)

col_list = c('date','true', 'week_ahead', 'point', 'point_avg')
res1 <- res1[, col_list]

######################### ------- save ---------#################################
res1[!complete.cases(res1),]
res1=res1[complete.cases(res1),]

write.csv(res1, paste0('../../Results/Point/forecast_', model_name,'_',mode,'.csv'), row.names = FALSE)


#plot：0，4，8
res11=res1[which(res1$week_ahead==0|
                   res1$week_ahead==4|
                   res1$week_ahead==8),]

true_data <- res11 %>%
  select(date, true) %>%
  mutate(week_ahead = "true")%>%
  distinct(date, .keep_all = TRUE) %>%
  rename(point = true)

res11$week_ahead=as.character(res11$week_ahead)
# Step 2: 与原始数据合并
df_combined <- bind_rows(res11[,c("date"  , "week_ahead", "point" )],
                         true_data[,c("date"  , "week_ahead", "point" )])


ggplot(df_combined, aes(x = date, y = point, color = week_ahead)) +
  geom_line() 







