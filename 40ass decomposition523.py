
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Import the ASX200 monthly data and create time series index based on the dates
month_data = pd.read_csv('ASX200Monthly.csv',usecols=['Date','Close'],index_col=['Date'])
month_data.index = pd.to_datetime(month_data.index)

#Plot the ASX200 monthly data including the titles and legend
plt.figure(figsize=(16,8))
plt.plot(month_data)
plt.title('ASX 200 price by Month', fontsize = 24) 
plt.xlabel('Time', fontsize = 18) 
plt.ylabel('ASX 200 price', fontsize = 18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
#plt.legend(loc='upper left', fontsize=18)


#Import the ASX200 daily data and create time series index based on the dates
#date_data = pd.read_csv('ASX200Daily.csv',usecols=['Date','Close'],index_col=['Date'])
date_data = pd.read_csv('AXJO.csv',usecols=['Date','Close'],index_col=['Date'])
#Remove Na values, set index and pick only data within the time range
date_data =date_data.dropna(axis=0)
date_data.index = pd.to_datetime(date_data.index)
date_data = date_data['2000-03-31':'2019-03-30']


#Calculate the value of M period
M=len(date_data)/19

#Create new series of daily closing prices so on any day where the stock market is not trading
#the index price is assumed to be the previous trading day's closing price
#k=[]
#
#for i in range(len(date_data)):    
#    if np.isnan(date_data['Close'][i])==True:
#        k.append(k[i-1])
#    else:
#        k.append(date_data['Close'][i])
#date_data['Adjusted Close']=k


#Plot the ASX200 daily data including the titles and legend
plt.figure(figsize=(16,8))
plt.plot(date_data['Close'])
plt.title('ASX 200 price by Day', fontsize = 24) 
plt.xlabel('Time', fontsize = 18) 
plt.ylabel('ASX 200 price', fontsize = 18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
#plt.legend(loc='upper left', fontsize=18)

month_data_diff = month_data.diff()
month_data_diff2 = month_data_diff.diff()

#Plot the ASX200 monthly data including the titles and legend for the order differencing
plt.figure(figsize=(16,8))
plt.plot(month_data_diff,'r-',label = "ASX first order diff")
plt.plot(month_data_diff2,'b-',label = "ASX second order diff")
plt.title('ASX 200 price change by Month Differencing', fontsize = 24) 
plt.xlabel('Time', fontsize = 18) 
plt.ylabel('Price change', fontsize = 18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
plt.legend(loc='upper left', fontsize=18)



daily_data_diff = date_data['Close'].diff()
daily_data_diff2 = daily_data_diff.diff()

#Plot the ASX200 daily data including the titles and legend for the order differencing
plt.figure(figsize=(16,8))
plt.plot(daily_data_diff,'r-',label = "ASX first order diff")
plt.plot(daily_data_diff2,'b-',label = "ASX second order diff")
plt.title('ASX 200 price change by Daily Differencing', fontsize = 24) 
plt.xlabel('Time', fontsize = 18) 
plt.ylabel('Price change', fontsize = 18)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
plt.legend(loc='upper left', fontsize=18)

#%% decomposition
daily_log=np.log(date_data)
plt.figure()
plt.plot(daily_log, color='green',label='daily log transformation')
plt.title('daily log transformation')
plt.xlabel('Year')
plt.ylabel('log price')
monthly_log=np.log(month_data)
plt.figure()
plt.plot(monthly_log, color='green',label='monthly log transformation')
plt.title('monthly log transformation')
plt.xlabel('Year')
plt.ylabel('log price')
#%% multiplicative Initial trend-cycle
#monthly data
Trend_cycle_m = monthly_log.rolling(12, center =True).mean().rolling(2,center = True).mean()
Trend_cycle_m=Trend_cycle_m.shift(-1)
plt.figure()
plt.plot(monthly_log, color='red',label='log')
plt.plot(Trend_cycle_m,color='green')
plt.title('Initial Trend_cycle estimate monthly(multiplicative)')
plt.xlabel('Year')
plt.ylabel('log price')

Trend_cycle_d = daily_log.rolling(20, center =True).mean().rolling(2,center = True).mean()
Trend_cycle_d=Trend_cycle_d.shift(-1)
plt.figure()
plt.plot(monthly_log, color='red',label='log')
plt.plot(Trend_cycle_d,color='green')
plt.title('Initial Trend_cycle estimate daily((multiplicative)M=20)')
plt.xlabel('Year')
plt.ylabel('log price')
#daily data
Trend_cycle_d1 = daily_log.rolling(250, center =True).mean().rolling(2,center = True).mean()
Trend_cycle_d1=Trend_cycle_d1.shift(-1)
plt.figure()
plt.plot(monthly_log, color='red',label='log')
plt.plot(Trend_cycle_d1,color='green')
plt.title('Initial Trend_cycle estimate daily((multiplicative)M=250)')
plt.xlabel('Year')
plt.ylabel('log price')

#%%
#calculate seasonal by removing trend-cycle
# If it is additive model, we shall use vs - r
Shat_monthly = monthly_log / Trend_cycle_m
# Showing the reasonable component
plt.figure()
plt.plot(Shat_monthly, '-r', label='Seasonal Approximation')
plt.title('Approximated Seasonal Component monthly(multiplicative)') 
plt.xlabel('Year')
plt.ylabel('Seasonal index')
#plt.legend(loc=2) 
#fig4.savefig('2019Lecture03_F04.png')
'''
Shat_dailyly = daily_log / Trend_cycle_d
# Showing the reasonable component
plt.figure()
plt.plot(Shat_dailyly, '-r', label='Seasonal Approximation')
plt.title('Approximated Seasonal Component daily(M=20)(multiplicative)') 
plt.xlabel('Year')
plt.ylabel('Seasonal index')
'''
Shat_dailyly1 = daily_log / Trend_cycle_d1
# Showing the reasonable component
plt.figure()
plt.plot(Shat_dailyly1, '-r', label='Seasonal Approximation')
plt.title('Approximated Seasonal Component daily(M=250)(multiplicative)') 
plt.xlabel('Year')
plt.ylabel('Seasonal index')
#%%
# monthly seasonal index
monthly_data1=Trend_cycle_m.iloc[11:-2]
monthly_sea = np.nan_to_num(monthly_data1)
#monthly_sea =monthly_sea.dropna(axis=0)
monthly_data_sea=np.reshape(monthly_sea,(18,12))
monthly_data_sea_avg=np.mean(monthly_data_sea[1:-1],axis=0)
c = 12/monthly_data_sea_avg.sum() #calculate the normalized constant
normalized_monthly=monthly_data_sea_avg*c
normalized_monthly=np.tile(normalized_monthly,18)
plt.figure()
plt.plot(normalized_monthly,color='green')
plt.title('Seasonal index monthly(multiplicative)')
plt.xlabel('Month')
plt.ylabel('Seasonal index')

# daily seasonal index   
daily_data1=Trend_cycle_d1.iloc[49:] #??????????????????????
daily_sea = np.nan_to_num(daily_data1)
daily_data_sea=np.reshape(daily_sea,(19,250))
##monthly_sea =monthly_sea.dropna(axis=0)
#daily_data_sea=np.reshape(daily_sea,(18,12))
daily_data_sea_avg=np.mean(daily_data_sea[1:-1],axis=0)
c = 250/daily_data_sea_avg.sum() #calculate the normalized constant
normalized_daily=daily_data_sea_avg*c
normalized_daily=np.tile(normalized_daily,19)
plt.figure()
plt.plot(normalized_daily,color='green')
plt.title('Seasonal index daily(M=250)(multiplicative)')
plt.xlabel('Day')
plt.ylabel('Seasonal index')



#%%addictive monthly
'''1 CALCULATE INITIAL TREND-CYCLE ESTIMATION BY MOVING AVERAGE'''
ts_monthly = monthly_log-Trend_cycle_m
plt.figure()
plt.plot(ts_monthly, '-r', label='Seasonal Approximation')
plt.title('Seasonal Component monthly(addictive)') 
plt.xlabel('Year')
plt.ylabel('Seasonal index')
'''2 CALCULATE SEASONAL INDEX'''
monthly_data1=ts_monthly.iloc[11:-2]
ts_monthly_zero = np.nan_to_num(monthly_data1)
monthly_S = np.reshape(ts_monthly_zero, (18,12))
monthly_avg = np.mean(monthly_S[0:17,:], axis=0)
#print(monthly_S.size)
#print(monthly_avg.size)

mean_allmonth = monthly_avg.mean()
monthly_avg_normalized = monthly_avg - mean_allmonth
print('monthly seasonal index:',monthly_avg_normalized.mean())
#%% daily
'''1 CALCULATE INITIAL TREND-CYCLE ESTIMATION BY MOVING AVERAGE'''
ts_daily = daily_log-Trend_cycle_d1
plt.figure()
plt.plot(ts_daily, '-r', label='Seasonal Approximation')
plt.title('Seasonal Component daily(addictive)') 
plt.xlabel('Year')
plt.ylabel('Seasonal index')
'''2 CALCULATE SEASONAL INDEX'''
daily_data1=ts_daily.iloc[49:]
ts_daily_zero = np.nan_to_num(daily_data1)
daily_S = np.reshape(ts_daily_zero, (19,250))
daily_avg = np.mean(daily_S[1:18,:], axis=0)
#print(monthly_S.size)
#print(monthly_avg.size)
mean_allday = daily_avg.mean()
daily_avg_normalized = daily_avg - mean_allday
print('daily seasonal index:',daily_avg_normalized.mean())

#%% monthly
'''3 CALCULATE THE SEASONAL ADJUSTED DATA'''
tiled_avg_m = np.tile(monthly_avg_normalized,18)
#monthly_data1=pd.DataFrame(monthly_data1,dtype=np.float)
#monthly_data2=pd.Series(monthly_data1)
monthly_log1=monthly_log.iloc[11:-2]
tiled_avg_m=pd.DataFrame(tiled_avg_m,index=monthly_log1.index)
seasonally_adjusted_m = monthly_log1 - tiled_avg_m.values
#plt.figure(figsize=(16,8))
fig, ax = plt.subplots(4, 1)
ax[0].plot(monthly_log1)
ax[1].plot(Trend_cycle_m)
ax[2].plot(ts_monthly)
ax[3].plot(seasonally_adjusted_m)
ax[0].legend(['log'], loc=2)
ax[1].legend(['trend'], loc=2)
ax[2].legend(['seasonality'], loc=2)
ax[3].legend(['seasonal adjusted monthly'], loc=2)

'''4 UPDATE THE TREND-CYCLE'''
#seasonally_adjusted_m=pd.Series(seasonally_adjusted_m)
T_final_m = seasonally_adjusted_m.rolling(12, center=True).mean().rolling(2, center = True).mean()
plt.figure()
plt.plot(T_final_m)
plt.title('Final TREND estimation monthly')
plt.xlabel('Month')
plt.ylabel('Number')


#%%daily

'''3 CALCULATE THE SEASONAL ADJUSTED DATA'''
tiled_avg_d = np.tile(daily_avg_normalized,19)
#monthly_data1=pd.DataFrame(monthly_data1,dtype=np.float)
#monthly_data2=pd.Series(monthly_data1)
daily_log1=daily_log.iloc[49:]
tiled_avg_d=pd.DataFrame(tiled_avg_d,index=daily_log1.index)
seasonally_adjusted_d = daily_log1 - tiled_avg_d.values
#plt.figure(figsize=(16,8))
fig, ax = plt.subplots(4, 1)
ax[0].plot(daily_log1)
ax[1].plot(Trend_cycle_d)
ax[2].plot(ts_daily)
ax[3].plot(seasonally_adjusted_d)
ax[0].legend(['log'], loc=2)
ax[1].legend(['trend'], loc=2)
ax[2].legend(['seasonality'], loc=2)
ax[3].legend(['seasonal adjusted daily'], loc=2)

'''4 UPDATE THE TREND-CYCLE'''
#seasonally_adjusted_d=pd.Series(seasonally_adjusted_d)
T_final_d = seasonally_adjusted_d.rolling(250, center=True).mean().rolling(2, center = True).mean()
plt.figure()
plt.plot(T_final_d)
plt.title('Final TREND estimation daily')
plt.xlabel('Day')
plt.ylabel('Number')
