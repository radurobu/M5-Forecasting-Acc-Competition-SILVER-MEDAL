# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:08:35 2020

@author: Radu

References:
    https://www.kaggle.com/ragnar123/very-fst-model
    Our val rmse score is 2.12543780258874
    TO DO: add 3 days to calendar
    LB: 1.01006
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
#import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
import lightgbm as lgb
#import dask_xgboost as xgb
#import dask.dataframe as dd
from sklearn import preprocessing, metrics
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
path = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy\Data/"
        

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# function to read the data and merge it (ignoring some columns, this is a very fst model)


def read_data():
    print('Reading files...')
    calendar = pd.read_csv(f'{path}calendar.csv')
    calendar['d_aux'] = calendar['d'].map(lambda x: int(x[2:])) 
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(f'{path}sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(f'{path}sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(f'{path}sample_submission.csv')
    return calendar, sell_prices, sales_train_validation, submission


########### Aggreate Data ##########    
calendar, sell_prices, sales_train_validation, submission = read_data()

# melt sales data, get it ready for training
sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
sales_train_validation = reduce_mem_usage(sales_train_validation)

SEE = sales_train_validation.tail(10000)

data_agg = sales_train_validation.groupby(['store_id', 'dept_id', 'day'])['demand'].agg('sum').reset_index()
data_agg['day_num'] = data_agg['day'].map(lambda x: int(x[2:]))
# Create pseudo dataset for exogenous vars used later in forecasting
data_agg_aux=pd.DataFrame(data_agg.iloc[:1,:].copy()) 
aux=pd.DataFrame(data_agg.iloc[:1,:].copy()) 
for s in data_agg['store_id'].unique():
    for d in data_agg['dept_id'].unique():
        for i in range(1914, 1942 + 3): #adding 3 days for lagged varaibles in exog data
            aux['store_id'] = s             
            aux['dept_id'] = d 
            aux['day'] = 'd_' + str(i)
            aux['demand'] = np.nan
            aux['day_num'] = i
            data_agg_aux = data_agg_aux.append(aux)
data_agg_aux = data_agg_aux.iloc[1:,:].reset_index(drop=True)
data_agg= data_agg.append(data_agg_aux)
data_agg = data_agg.sort_values(by=['store_id', 'dept_id', 'day_num'])
data_agg = pd.merge(data_agg, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
data_agg.drop(['d', 'day', 'd_aux'], inplace = True, axis = 1)
data_agg['date'] = pd.to_datetime(data_agg['date'])
data_agg['mday'] = getattr(data_agg["date"].dt, 'day').astype("int16")
data_agg['week'] = getattr(data_agg["date"].dt, 'weekofyear').astype("int16")

del sales_train_validation 
del sell_prices
#del submission
#####################  

print(data_agg['store_id'].unique())
print(data_agg['dept_id'].unique())
        
############################
#   Add Exogenous Data     #
###########################

#VAR_Data = pd.merge(VAR_Data, data_agg, how='left', left_on=['store_id', 'dept_id', 'day_num'], right_on=['store_id', 'dept_id', 'day_num']) 
      
VAR_Data = data_agg.copy()

conditions = [
        ('CA' == VAR_Data['store_id'].map(lambda x: str(x[:2]))),
        ('TX' == VAR_Data['store_id'].map(lambda x: str(x[:2]))),
        ('WI' == VAR_Data['store_id'].map(lambda x: str(x[:2])))] 

choices = [VAR_Data['snap_CA'], VAR_Data['snap_TX'], VAR_Data['snap_WI']]

VAR_Data['snap'] = np.select(conditions, choices, default=0)

# Missing value assignment
miss_coll = ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']
for i in miss_coll:
    VAR_Data[i] = VAR_Data[i].fillna('none')


#Create Lagged variables for exog vars
VAR_Data['L1_event_type_1'] =VAR_Data.groupby(['store_id', 'dept_id'])['event_type_1'].shift(-1)
VAR_Data['L1_event_name_1'] =VAR_Data.groupby(['store_id', 'dept_id'])['event_name_1'].shift(-1)
VAR_Data['L2_event_type_1'] =VAR_Data.groupby(['store_id', 'dept_id'])['event_type_1'].shift(-2)
VAR_Data['L2_event_name_1'] =VAR_Data.groupby(['store_id', 'dept_id'])['event_name_1'].shift(-2)
VAR_Data['L3_event_type_1'] =VAR_Data.groupby(['store_id', 'dept_id'])['event_type_1'].shift(-3)
VAR_Data['L3_event_name_1'] =VAR_Data.groupby(['store_id', 'dept_id'])['event_name_1'].shift(-3)

#VAR_Data['L1_demand'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(1)
#VAR_Data['L2_demand'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(2)
#VAR_Data['L3_demand'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(3)
VAR_Data['L7_demand'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(7)
VAR_Data['L14_demand'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(14)
VAR_Data['L28_demand'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(28)

#VAR_Data['mean_demand_7_7'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x: x.rolling(7).mean())
#VAR_Data['mean_demand_7_28'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x: x.rolling(28).mean())
VAR_Data['mean_demand_28_7'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(28).transform(lambda x: x.rolling(7).mean())
VAR_Data['mean_demand_28_28'] = VAR_Data.groupby(['store_id', 'dept_id'])['demand'].shift(28).transform(lambda x: x.rolling(28).mean())


######################################
#Creata dataset for Level Model (GBM)#
######################################

# Keep Columns for GBM Estimation
GBM_index_vars = ['year', 'day_num', 'date', 'wm_yr_wk', 'weekday']
GBM_train_vars = ['store_id', 'dept_id', 'wday', 'week', 'mday', 'month', 'event_name_1', 'event_type_1', 'event_name_2',
                     'event_type_2', 'snap', 'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1',
                     'L3_event_type_1', 'L3_event_name_1', 'L7_demand', 'L14_demand', 
                     'L28_demand', 'mean_demand_28_7', 'mean_demand_28_28']
GBM_target = ['demand']

GBM_Data = VAR_Data[ list(GBM_index_vars + GBM_train_vars + GBM_target)]
GBM_Data = GBM_Data[GBM_Data['day_num'] <= 1913] #Change for submission

# Train Test Split (test will be the submission sample)
valid_samples = {
        "end_sample5":1913,
        "start_sample5":1886,
        "end_sample4":1859,
        "start_sample4":1832,
        "end_sample3":1805,
        "start_sample3":1778,
        "end_sample2":1751,
        "start_sample2":1724,
        "end_sample1":1697,
        "start_sample1":1670
        }

valid_ids=pd.DataFrame()
for i in range(1,6):
    range_list = list(range(valid_samples[f"start_sample{i}"], valid_samples[f"end_sample{i}"] + 1))
    valid_ids_aux = GBM_Data.index[GBM_Data['day_num'].isin(range_list)].tolist()
    valid_ids = valid_ids.append(valid_ids_aux)

GBM_Data_train = GBM_Data[(~GBM_Data.index.isin(valid_ids[0].values))][GBM_train_vars]
GBM_Data_valid = GBM_Data[(GBM_Data.index.isin(valid_ids[0].values))][GBM_train_vars]
GBM_y_train = GBM_Data[(~GBM_Data.index.isin(valid_ids[0].values))][GBM_target]
GBM_y_valid = GBM_Data[(GBM_Data.index.isin(valid_ids[0].values))][GBM_target]
GBM_sub = VAR_Data[ (VAR_Data['day_num'] >= 1914) & (VAR_Data['day_num'] <= 1941)][GBM_train_vars] #Change for submission

# Label Encoding 
encod_cols = ['store_id', 'dept_id','event_name_1', 'event_name_1', 'event_type_1', 'event_name_2',
             'event_type_2', 'snap', 'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1',
             'L3_event_type_1', 'L3_event_name_1']
for col in encod_cols:
    le = preprocessing.LabelEncoder()
    GBM_Data_train[col] = le.fit_transform(GBM_Data_train[col])
    GBM_Data_valid[col] = le.transform(GBM_Data_valid[col])
    GBM_sub[col] = le.transform(GBM_sub[col])

###################
# LGBM Estimation #
###################

# Train dataset
cat_feats = ['store_id', 'dept_id', 'event_name_1', 'event_type_1', 'event_name_2',
             'event_type_2', 'snap', 'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1',
             'L3_event_type_1', 'L3_event_name_1']

y_pred_val = pd.DataFrame()
y_pred_sub = pd.DataFrame()
print('LGBM Estimation Start: {}'.format(datetime.now())) 
        
train_data = lgb.Dataset(GBM_Data_train, label = GBM_y_train, categorical_feature=cat_feats, free_raw_data=False)
valid_data = lgb.Dataset(GBM_Data_valid, label = GBM_y_valid, categorical_feature=cat_feats, free_raw_data=False)

params = {
        "objective" : "poisson",
        "metric" :"rmse",
#        "force_row_wise" : True,
        "learning_rate" : 0.03,
        "early_stopping_round" : 200,
#         "sub_feature" : 0.8,
        "num_leaves" : 31,
#        "bagging_fraction" : 0.5,
#        "bagging_freq" : 10,
#        "lambda_l2" : 0.1,
#         "nthread" : 4
        'verbosity': 0,
        'num_iterations' : 10000,
}
 
m_lgb = lgb.train(params, train_data, valid_sets = [valid_data], verbose_eval=100)

y_pred =pd.DataFrame(m_lgb.predict(GBM_Data_valid))

SEE = pd.merge(GBM_Data[(GBM_Data.index.isin(valid_ids[0].values))].reset_index(drop=True), y_pred, how='left', right_index=True, left_index=True)
#92.6962
######################
#  Feature Imprtance #
######################
feature_imp = pd.DataFrame(sorted(zip(m_lgb.feature_importance(),GBM_Data_train.columns)), columns=['Value','Feature'])
   
##########################################
# Predict Validation sample (itterative) #
##########################################

#GBM_Data_valid[['L1_level', 'L2_level', 'L3_level', 'L7_level', 'L14_level']] = np.nan
y_pred_val=pd.DataFrame()
for s in GBM_Data_valid['store_id'].unique():
    for d in GBM_Data_valid['dept_id'].unique():
        print(f'Estimation at {s} {d}')
        
        GBM_Data_valid_aux = GBM_Data_valid[(GBM_Data_valid['store_id'] == s) & (GBM_Data_valid['dept_id'] == d)].reset_index(drop=True)
        y_pred_aux=pd.DataFrame()
        
        for t in range(0, len(GBM_Data_valid_aux)):
            obs = GBM_Data_valid_aux.iloc[t,:].astype('float32')
            estimate = m_lgb.predict(obs)
            y_pred_aux.loc[t,'demand_fcast'] = estimate
            '''
            GBM_Data_valid_aux.loc[t+1, 'L1_demand'] = estimate
            GBM_Data_valid_aux.loc[t+2, 'L2_demand'] = estimate
            GBM_Data_valid_aux.loc[t+3, 'L3_demand'] = estimate'''
            GBM_Data_valid_aux.loc[t+7, 'L7_demand'] = estimate
            GBM_Data_valid_aux.loc[t+14, 'L14_demand'] = estimate
        y_pred_val = y_pred_val.append(y_pred_aux)

            
df_forecast_val = pd.merge(GBM_Data[(GBM_Data.index.isin(valid_ids[0].values))].reset_index(drop=True), y_pred_val.reset_index(drop=True), how='left', right_index=True, left_index=True)      

conditions = [
         ( (1670 <= df_forecast_val['day_num']) & ( df_forecast_val['day_num'] <= 1697)),
         ( (1724 <= df_forecast_val['day_num']) & ( df_forecast_val['day_num'] <= 1751)),
         ( (1778 <= df_forecast_val['day_num']) & ( df_forecast_val['day_num'] <= 1805)),
         ( (1832 <= df_forecast_val['day_num']) & ( df_forecast_val['day_num'] <= 1859)),
         ( (1886 <= df_forecast_val['day_num']) & ( df_forecast_val['day_num'] <= 1913))] 

choices = [5,4,3,2,1]

df_forecast_val['valid_sample_nr'] = np.select(conditions, choices, default=0)

            
##########
#  WRMSE  #
##########
path_results = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy/"
weights_csv = pd.read_csv(f'{path_results}weights_csv.csv')

from sklearn.metrics import mean_squared_error
from math import sqrt
for i in range(1,6):
    RMSE_ALL = 0
    for s, d, w in list(zip(weights_csv['store_id'], weights_csv['dept_id'], weights_csv['dollar_sales'])):
        RMSE_true = df_forecast_val[(df_forecast_val['valid_sample_nr']==i) & (df_forecast_val['store_id']==s) & (df_forecast_val['dept_id']==d)].demand
        RMSE_pred = df_forecast_val[(df_forecast_val['valid_sample_nr']==i) & (df_forecast_val['store_id']==s) & (df_forecast_val['dept_id']==d)].demand_fcast
        RMSE = sqrt(mean_squared_error(RMSE_true, RMSE_pred))
        #print(f"RMSE {s} {d}: {RMSE}")
        RMSE_ALL += sqrt(mean_squared_error(RMSE_true, RMSE_pred)) * w
    print(f'WRMSE Valid Sampre {i}: {RMSE_ALL}')  
'''
WRMSE Valid Sampre 1: 139.29895778207273
WRMSE Valid Sampre 2: 165.464837008929
WRMSE Valid Sampre 3: 132.79073403262626
WRMSE Valid Sampre 4: 144.32074954682437
WRMSE Valid Sampre 5: 112.04790008266816
'''


##########################################
# Predict Submission sample (itterative) #
##########################################
y_pred_sub=pd.DataFrame()
for s in GBM_sub['store_id'].unique():
    for d in GBM_sub['dept_id'].unique():
        print(f'Estimation at {s} {d}')
        
        GBM_sub_aux = GBM_sub[(GBM_sub['store_id'] == s) & (GBM_sub['dept_id'] == d)].reset_index(drop=True)
        y_pred_aux=pd.DataFrame()
        
        for t in range(0, len(GBM_sub_aux)):
            obs = GBM_sub_aux.iloc[t,:].astype('float32')
            estimate = m_lgb.predict(obs)
            y_pred_aux.loc[t,'demand_fcast'] = estimate
            '''
            GBM_sub_aux.loc[t+1, 'L1_demand'] = estimate
            GBM_sub_aux.loc[t+2, 'L2_demand'] = estimate
            GBM_sub_aux.loc[t+3, 'L3_demand'] = estimate'''
            GBM_sub_aux.loc[t+7, 'L7_demand'] = estimate
            GBM_sub_aux.loc[t+14, 'L14_demand'] = estimate
        y_pred_sub = y_pred_sub.append(y_pred_aux)

df_forecast_sub = pd.merge(VAR_Data[ (VAR_Data['day_num'] >= 1914) & (VAR_Data['day_num'] <= 1941)].reset_index(drop=True), y_pred_sub.reset_index(drop=True), how='left', right_index=True, left_index=True)      

            
###############
#  Submission #
###############

#Calculate weights
def read_data():
    print('Reading files...')
    calendar = pd.read_csv(f'{path}calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(f'{path}sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(f'{path}sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(f'{path}sample_submission.csv')
    return calendar, sell_prices, sales_train_validation, submission

cal_df, price_df, df, sub = read_data()

cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))
price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"

for day in tqdm(range(1885, 1914)): # Change here for second submission
    wk_id = list(cal_df[cal_df["d"]==day]["wm_yr_wk"])[0]
    wk_price_df = price_df[price_df["wm_yr_wk"]==wk_id]
    df = df.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')
    df["unit_sales_" + str(day)] = df["sell_price"] * df["d_" + str(day)]
    df.drop(columns=["sell_price"], inplace=True)
    
df["dollar_sales"] = df[[c for c in df.columns if c.find("unit_sales")==0]].sum(axis=1)
df.drop(columns=[c for c in df.columns if c.find("unit_sales")==0], inplace=True)
grouped_dollar_sales = df.groupby(['store_id', 'dept_id'])["dollar_sales"].sum().reset_index().rename(columns = {"dollar_sales":"grouped_dollar_sales"})
df = df.merge(grouped_dollar_sales, how='left', left_on=['store_id', 'dept_id'], right_on=['store_id', 'dept_id'])
df["weight"] = df["dollar_sales"] / df["grouped_dollar_sales"]
df = df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weight']]

print(df.groupby(['store_id', 'dept_id']).sum())

#Prepare submission file
sub = pd.DataFrame(sub['id'])
aux = sub['id'].str.split("_", 6, expand=True)
sub['item_id'] = aux[0] + "_" + aux[1] + "_" + aux[2]
sub['dept_id'] = aux[0] + "_" + aux[1]
sub['store_id'] = aux[3] + "_" + aux[4]
sub['state_id'] = aux[3]

# Add weights to submission
sub = sub.merge(df[['item_id', 'store_id', 'weight']], how='left', left_on=['item_id', 'store_id'], right_on=['item_id', 'store_id'])

# Add forecasts to submission
sub = sub.merge(df_forecast_sub[['store_id', 'dept_id', 'day_num', 'demand_fcast']], how='left', left_on=['store_id', 'dept_id'], right_on=['store_id', 'dept_id'])
sub['demand_submission'] = sub['demand_fcast'] * sub['weight']

# Transpose
sub = sub.pivot(index='id', columns='day_num', values='demand_submission').reset_index()
sub.columns = submission.columns

path_results = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy/"
sub.to_csv(f'{path_results}submission.csv', index=False)
           
            
            
    
    
    
    
    
    
    
    
    
    
    
############################
# Start of estimation loop #
############################ 
print('Estimation & Forecast Start: {}'.format(datetime.now()))  
first_store = True
for store_est in VAR_Data['store_id'].unique():
    VAR_Data_state = VAR_Data[VAR_Data['store_id'] == store_est]
    print(store_est)
    
    #######################################
    #    Reshape Data for VAR Modeling    #
    ######################################
    
    stores = VAR_Data_state['store_id'].unique()
    depts = VAR_Data_state['dept_id'].unique()
    
    VAR_Data_aux = pd.DataFrame()
    
    first = True
    for s in stores:
        for d in depts:
            VAR_Data_aux =  VAR_Data_state.loc[(VAR_Data_state['store_id'] == s) & (VAR_Data_state['dept_id'] == d) , ['store_id', 'dept_id', 'date', 'day_num', 'd_level']].reset_index(drop=True)
            if first:
                VAR_Data_t = pd.DataFrame(VAR_Data_aux)
                VAR_Data_t = VAR_Data_t.rename(columns = {'d_level': 'd_level_{}_{}'.format(s,d)})
                first = False
            else:
                VAR_Data_t = VAR_Data_t.merge(VAR_Data_aux['d_level'], how = 'left' , left_index = True, right_index = True)
                VAR_Data_t = VAR_Data_t.rename(columns = {'d_level': 'd_level_{}_{}'.format(s,d)})
    
    # Bring exogenous variables
    VAR_Data_t = VAR_Data_t.merge(VAR_Data_state[['store_id', 'dept_id', 'day_num','event_type_1_Cultural', 'event_type_1_National', 
                                            'event_type_1_Religious', 'event_type_1_Sporting', 'snap',
                                            'month_2', 'month_3','month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
                                           'month_10', 'month_11', 'month_12', 'L1_event_type_1_Cultural',
                                           'L2_event_type_1_Cultural', 'L3_event_type_1_Cultural',
                                           'L1_event_type_1_National', 'L2_event_type_1_National',
                                           'L3_event_type_1_National', 'L1_event_type_1_Religious',
                                           'L2_event_type_1_Religious', 'L3_event_type_1_Religious',
                                           'L1_event_type_1_Sporting', 'L2_event_type_1_Sporting',
                                           'L3_event_type_1_Sporting']
                                            ], how='left', right_on = ['store_id', 'dept_id', 'day_num'], left_on = ['store_id', 'dept_id', 'day_num'])
    
    
    #########################
    #    Train Test Split  #
    ########################  
    # first date: 2011-01-29
    # last date: 2016-05-22
    #Validation: 
    #Test:   2016-04-25 -  2016-05-25   
    #Train:  2011-01-30 -  2016-04-24

    
    #VAR_Data_t_Train = VAR_Data_t[VAR_Data_t['date'] <= '2016-02-28']
    VAR_Data_t_Train = VAR_Data_t[(VAR_Data_t['date'] >= '2011-01-30') & (VAR_Data_t['date'] <= '2016-04-24')].reset_index(drop=True)
    VAR_Data_t_Test  = VAR_Data_t[(VAR_Data_t['date'] >= '2016-04-25') & (VAR_Data_t['date'] <= '2016-05-22')].reset_index(drop=True) #Change here for next sub
    #VAR_Data_t_Valid = VAR_Data_t[VAR_Data_t['date'] >= '2016-03-28']
    
    ###########################
    #    Estimate VAR Model  #
    ##########################       
    from statsmodels.tsa.api import VAR
    
    
    endog = VAR_Data_t_Train.iloc[:, 4:11].astype(float)
    endog_index = VAR_Data_t_Train.loc[:, ['store_id', 'date', 'day_num']]
    exog  = VAR_Data_t_Train.iloc[:, 11:].astype(float)
    
    model = VAR(endog, exog=exog)           
    model_fitted = model.fit(28)
    #model_fitted.summary()
    
    # Plot Imput vars
    #model_fitted.plot()
    # Plot Forecasts
    #model_fitted.plot_forecast(28, exog_future = exog_future)
    # Plot IRF
    #irf = model_fitted.irf(10)
    #irf.plot(orth=False)
    #irf.plot(impulse='d_level_CA_FOODS_3', orth=False)

    #########################
    #  Forecast VAR model  #
    ########################  
    #model_fitted = model.fit(1)
    # Get the lag order
    lag_order = model_fitted.k_ar
    print(lag_order)  
    
    #start of itteriative forcasting loop
    first_day_itter = True
    for p_begin in range(0, 1913 - lag_order*2): # since lag_oreder = forcast horizont in this setup
        p_stop = p_begin + lag_order # since lag_oreder = forcast horizont in this setup
        
        # Input data for forecasting
        forecast_input = endog.iloc[p_begin:p_stop].values
        forecast_input_index = endog_index.iloc[p_begin:p_stop]
        forecast_exog = exog.iloc[p_begin:p_stop].values
        
        
        forecast_output_index = pd.DataFrame(columns = forecast_input_index.columns, index = list(range(forecast_input_index.index[-1] + 1, forecast_input_index.index[-1] + 29) ))
        #forecast_output_index['date'] = pd.to_datetime(forecast_output_index['date'] )
        #forecast_output_index['day_num'] =forecast_output_index['day_num'].astype(float)
        
        for i in range(1, 29):
            forecast_output_index.loc[[forecast_input_index.index[-1] + i] ,'date'] = forecast_input_index.loc[forecast_input_index.index[-1], 'date'] + pd.Timedelta(days=i)
            forecast_output_index.loc[[forecast_input_index.index[-1] + i] ,'day_num'] = int(forecast_input_index.tail(1).day_num + i)
            forecast_output_index.loc[[forecast_input_index.index[-1] + i] ,'store_id'] = forecast_input_index.loc[forecast_input_index.index[-1], 'store_id']
        forecast_output_index['fcast_itter_startday'] = p_stop + 2
        
        # Forecast
        nobs = 28
        fc = model_fitted.forecast(y=forecast_input, steps=nobs, exog_future=forecast_exog)
        df_forecast = pd.DataFrame(fc, index=forecast_output_index.index, columns=endog.columns + '_fcast')
        
        # Calculate Forecasted values
        last_day = pd.DataFrame(VAR_Data_state[(VAR_Data_state['store_id'] == store_est) & (VAR_Data_state['day_num'] == p_stop + 1) ].level)
        last_day = last_day.values.transpose()
        last_day = pd.DataFrame(last_day, index=[p_stop - 1], columns=df_forecast.columns)
        df_forecast = last_day.append(df_forecast)
        
        # Predicted values
        df_forecast = df_forecast.expanding(1).sum()
        
        if first_day_itter:
            df_forecast_all = pd.merge(forecast_output_index.reset_index(), calendar[['d_aux', 'wday']], how='left', left_on='day_num', right_on='d_aux').set_index('index')
            df_forecast_all = df_forecast_all.drop(columns = ['d_aux'])
            df_forecast_all = pd.merge(df_forecast_all, df_forecast.iloc[1:, :], how='left', left_index=True, right_index=True)
            first_day_itter = False
        else:
            df_forecast_all_aux = pd.merge(forecast_output_index.reset_index(), calendar[['d_aux', 'wday']], how='left', left_on='day_num', right_on='d_aux').set_index('index')
            df_forecast_all_aux = df_forecast_all_aux.drop(columns = ['d_aux'])
            df_forecast_all_aux = pd.merge(df_forecast_all_aux, df_forecast.iloc[1:, :], how='left', left_index=True, right_index=True)
            df_forecast_all = df_forecast_all.append(df_forecast_all_aux)

    #####
    if first_store:
        agg_forecast_t = df_forecast_all.copy().reset_index(drop=True)
        #first_store = False
    else:
        agg_forecast_t = pd.merge(agg_forecast_t, df_forecast_all.drop(columns = ['store_id', 'date', 'day_num', 'wday', 'fcast_itter_startday']).reset_index(drop=True), how='left', left_index=True, right_index=True)


     ####################################
     #  Forecast Test(Submison sample)  #
     ####################################

    forecast_input_sub = endog.values[-lag_order:]
    forecast_input_index_sub = endog_index[-lag_order:]
    exog_future = VAR_Data_t_Test.iloc[:, 11:].astype(float)
    
    forecast_output_index_sub = pd.DataFrame(columns = forecast_input_index_sub.columns, index = list(range(forecast_input_index_sub.index[-1] + 1, forecast_input_index_sub.index[-1] + 29) ))

    for i in range(1, 29):
        forecast_output_index_sub.loc[[forecast_input_index_sub.index[-1] + i] ,'date'] = forecast_input_index_sub.loc[forecast_input_index_sub.index[-1], 'date'] + pd.Timedelta(days=i)
        forecast_output_index_sub.loc[[forecast_input_index_sub.index[-1] + i] ,'day_num'] = int(forecast_input_index_sub.tail(1).day_num + i)
        forecast_output_index_sub.loc[[forecast_input_index_sub.index[-1] + i] ,'store_id'] = forecast_input_index_sub.loc[forecast_input_index_sub.index[-1], 'store_id']
     
    # Forecast
    nobs = 28
    fc_sub = model_fitted.forecast(y=forecast_input_sub, steps=nobs, exog_future=exog_future)
    df_forecast_sub = pd.DataFrame(fc_sub, index=forecast_output_index_sub.index, columns=endog.columns + '_fcast')
    
    # Calculate Forecasted values
    last_day_sub = pd.DataFrame(VAR_Data_state[(VAR_Data_state['store_id'] == store_est) & (VAR_Data_state['day_num'] == 1913) ].level) #Change here for next sub
    last_day_sub = last_day_sub.values.transpose()
    last_day_sub = pd.DataFrame(last_day_sub, index=[1911], columns=df_forecast_sub.columns) #Change here for next sub
    df_forecast_sub = last_day_sub.append(df_forecast_sub)
    
    # Predicted values
    df_forecast_sub = df_forecast_sub.expanding(1).sum()

    #####
    if first_store:
        agg_forecast_t_sub = pd.merge(forecast_output_index_sub.reset_index(), calendar[['d_aux', 'wday']], how='left', left_on='day_num', right_on='d_aux').set_index('index')
        agg_forecast_t_sub = agg_forecast_t_sub.drop(columns = ['d_aux'])
        agg_forecast_t_sub = pd.merge(agg_forecast_t_sub, df_forecast_sub.iloc[1:, :], how='left', left_index=True, right_index=True)
        first_store = False
    else:
        agg_forecast_t_aux_sub = pd.merge(forecast_output_index_sub.reset_index(), calendar[['d_aux', 'wday']], how='left', left_on='day_num', right_on='d_aux').set_index('index')
        agg_forecast_t_aux_sub = agg_forecast_t_aux_sub.drop(columns = ['d_aux'])
        agg_forecast_t_aux_sub = pd.merge(agg_forecast_t_aux_sub, df_forecast_sub.iloc[1:, :], how='left', left_index=True, right_index=True)
        agg_forecast_t_sub = pd.merge(agg_forecast_t_sub, agg_forecast_t_aux_sub.drop(columns = ['store_id', 'date', 'day_num', 'wday']), how='left', left_index=True, right_index=True)
print('Estimation & Forecast End: {}'.format(datetime.now()))  
#end of loop

#path_results = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy/"
#agg_forecast_t.to_csv(f'{path_results}agg_forecast_t_corect.csv', index=False)
#agg_forecast_t_sub.to_csv(f'{path_results}agg_forecast_t_sub.csv', index=False)
#agg_forecast_t = pd.read_csv(f'{path_results}agg_forecast_t_corect.csv')
#agg_forecast_t_sub = pd.read_csv(f'{path_results}agg_forecast_t_sub.csv')

#Transpose Train  
first = True
for col in agg_forecast_t.columns.drop(['store_id', 'date', 'day_num', 'wday', 'fcast_itter_startday']):
    if first:
        agg_forecast = agg_forecast_t[['store_id', 'date', 'day_num', 'wday', 'fcast_itter_startday', col]].rename(columns = {col: 'level_fcast'})
        agg_forecast['dept_id'] = col.split("_",6)[4] + "_" + col.split("_",6)[5]
        first= False
    else:
        aux = agg_forecast_t[['store_id', 'date', 'day_num', 'wday', 'fcast_itter_startday', col]].rename(columns = {col: 'level_fcast'})
        aux['store_id'] = col.split("_",6)[2] + "_" + col.split("_",6)[3]
        aux['dept_id'] = col.split("_",6)[4] + "_" + col.split("_",6)[5]
        agg_forecast = agg_forecast.append(aux)
agg_forecast = agg_forecast[['store_id', 'dept_id', 'fcast_itter_startday', 'date', 'day_num', 'wday', 'level_fcast']]

agg_forecast = agg_forecast.merge(mean_season, how= 'left', left_on=['store_id', 'dept_id', 'wday'], right_on=['store_id', 'dept_id', 'wday'])
agg_forecast['demand_fcast'] = agg_forecast['level_fcast'] + agg_forecast['season']

SEE = agg_forecast.tail(1000)

#Transpose Test(Submission)
first = True
for col in agg_forecast_t_sub.columns.drop(['store_id', 'date', 'day_num', 'wday']):
    if first:
        agg_forecast_sub = agg_forecast_t_sub[['store_id', 'date', 'day_num', 'wday', col]].rename(columns = {col: 'level_fcast'})
        agg_forecast_sub['dept_id'] = col.split("_",6)[4] + "_" + col.split("_",6)[5]
        first= False
    else:
        aux_sub = agg_forecast_t_sub[['store_id', 'date', 'day_num', 'wday', col]].rename(columns = {col: 'level_fcast'})
        aux_sub['store_id'] = col.split("_",6)[2] + "_" + col.split("_",6)[3]
        aux_sub['dept_id'] = col.split("_",6)[4] + "_" + col.split("_",6)[5]
        agg_forecast_sub = agg_forecast_sub.append(aux_sub)
agg_forecast_sub = agg_forecast_sub[['store_id', 'dept_id', 'date', 'day_num', 'wday', 'level_fcast']]

agg_forecast_sub = agg_forecast_sub.merge(mean_season, how= 'left', left_on=['store_id', 'dept_id', 'wday'], right_on=['store_id', 'dept_id', 'wday'])
agg_forecast_sub['demand_fcast'] = agg_forecast_sub['level_fcast'] + agg_forecast_sub['season']

SEE = agg_forecast_sub.tail(1000)

#########################################
#Creata dataset for Residual Model (GBM)#
#########################################

data_agg_plus = data_agg.copy()

# Add features
conditions = [
        ('CA' == data_agg_plus['store_id'].map(lambda x: str(x[:2]))),
        ('TX' == data_agg_plus['store_id'].map(lambda x: str(x[:2]))),
        ('WI' == data_agg_plus['store_id'].map(lambda x: str(x[:2])))] 
choices = [data_agg_plus['snap_CA'], data_agg_plus['snap_TX'], data_agg_plus['snap_WI']]
data_agg_plus['snap'] = np.select(conditions, choices, default=0)

data_agg_plus['L1_event_type_1'] =data_agg_plus.groupby(['store_id', 'dept_id'])['event_type_1'].shift(-1)
data_agg_plus['L1_event_name_1'] =data_agg_plus.groupby(['store_id', 'dept_id'])['event_name_1'].shift(-1)
data_agg_plus['L2_event_type_1'] =data_agg_plus.groupby(['store_id', 'dept_id'])['event_type_1'].shift(-2)
data_agg_plus['L2_event_name_1'] =data_agg_plus.groupby(['store_id', 'dept_id'])['event_name_1'].shift(-2)
data_agg_plus['L3_event_type_1'] =data_agg_plus.groupby(['store_id', 'dept_id'])['event_type_1'].shift(-3)
data_agg_plus['L3_event_name_1'] =data_agg_plus.groupby(['store_id', 'dept_id'])['event_name_1'].shift(-3)

# Add Demand Lags Dmean_{lag}_{win}
#data_agg_plus['Demand_lag7'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7)
#data_agg_plus['Demand_lag14'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(14)
#data_agg_plus['Demand_lag28'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(28)
#data_agg_plus['Dmean_7_7'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(7).mean())
#data_agg_plus['Dmean_7_14'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(14).mean())
#data_agg_plus['Dmean_14_7'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(14).transform(lambda x : x.rolling(7).mean())
#data_agg_plus['Dmean_14_14'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(14).transform(lambda x : x.rolling(14).mean())
#data_agg_plus['Dmean_7_28'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(28).mean())
#data_agg_plus['dif_Dmean_7_7'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(7).mean()) - data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(14).mean())
#data_agg_plus['dif_Dmean_7_28'] = data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(7).mean()) - data_agg_plus.groupby(['store_id', 'dept_id'])['demand'].shift(7).transform(lambda x : x.rolling(28).mean())


SEE = data_agg_plus.head(10000)

df_resid = agg_forecast.merge(data_agg_plus[[ 'demand', 'store_id', 'dept_id', 'day_num',
                                        'month',  'event_name_1', 'event_type_1',
                                       'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1', 
                                       'L3_event_type_1', 'L3_event_name_1', 'event_name_2', 'event_type_2', 'snap']], 
                                      how='left', right_on = ['store_id', 'dept_id', 'day_num'], left_on = ['store_id', 'dept_id', 'day_num'])

df_resid_sub = agg_forecast_sub.merge(data_agg_plus[[ 'demand', 'store_id', 'dept_id', 'day_num',
                                        'month',  'event_name_1', 'event_type_1',
                                       'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1', 
                                       'L3_event_type_1', 'L3_event_name_1', 'event_name_2', 'event_type_2', 'snap']], 
                                      how='left', right_on = ['store_id', 'dept_id', 'day_num'], left_on = ['store_id', 'dept_id', 'day_num'])

# Missing value assignment
miss_coll = ['event_name_1', 'event_name_2', 'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1', 'L3_event_type_1', 'L3_event_name_1', 'event_type_1', 'event_type_2']
for i in miss_coll:
    df_resid[i] = df_resid[i].fillna('none')
    df_resid_sub[i] = df_resid_sub[i].fillna('none')

# Add Residual features 
df_resid['resid'] = df_resid['demand'] - df_resid['demand_fcast']
df_resid_sub['resid'] = np.nan

# Add Forecast Itteration
df_resid['fcast_itter'] =  df_resid['day_num'] - df_resid['fcast_itter_startday'] + 1
df_resid_sub['fcast_itter'] =  df_resid_sub['day_num'] - 1914 + 1 #Change here for submisson

SEE = df_resid.head(1000)

# Add Residual features
'''
df_resid['resid_lag1'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(1)
df_resid['resid_lag2'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(2)
df_resid['resid_lag3'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(3)
df_resid['resid_lag4'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(4)
df_resid['resid_lag5'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(5)
df_resid['resid_lag6'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(6)
df_resid['resid_lag7'] = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid'].shift(7)
df_resid["rmean_1_7"]  = df_resid.groupby(['store_id', 'dept_id', 'fcast_itter_startday'])['resid_lag1'].transform(lambda x : x.rolling(7).mean())'''

'''
df_resid_sub['resid_lag1'] = np.nan
df_resid_sub['resid_lag2'] = np.nan
df_resid_sub['resid_lag3'] = np.nan
df_resid_sub['resid_lag4'] = np.nan
df_resid_sub['resid_lag5'] = np.nan
df_resid_sub['resid_lag6'] = np.nan
df_resid_sub['resid_lag7'] = np.nan
df_resid_sub["rmean_1_7"]  = np.nan
'''

# Train Test Split (test will be the submission sample)
valid_samples = {
        "end_sample5":1913,
        "start_sample5":1886,
        "end_sample4":1859,
        "start_sample4":1832,
        "end_sample3":1805,
        "start_sample3":1778,
        "end_sample2":1751,
        "start_sample2":1724,
        "end_sample1":1697,
        "start_sample1":1670
        }

valid_ids=pd.DataFrame()
for i in range(1,6):
    range_list = list(range(valid_samples[f"start_sample{i}"], valid_samples[f"end_sample{i}"] + 1))
    valid_ids_aux = df_resid.index[df_resid['day_num'].isin(range_list)].tolist()
    valid_ids = valid_ids.append(valid_ids_aux)

df_resid_train = df_resid[(df_resid['fcast_itter_startday'] <= 1886) & (~df_resid.index.isin(valid_ids.values))] #Change here for Subbmison
#df_resid_train = df_resid[(df_resid['fcast_itter_startday'] >= 1160)&(df_resid['fcast_itter_startday'] <= 1885)] #Change here for Subbmison
#df_resid_test = df_resid[df_resid['fcast_itter_startday'] == 1858] #Change here for Subbmison
df_resid_valid = df_resid[(df_resid.index.isin(valid_ids.values))]

#####TO DELETE:
#df_resid_train = df_resid_train[(df_resid_train ['store_id'] == 'CA_3')&(df_resid_train ['dept_id'] == 'HOUSEHOLD_1')]
#df_resid_valid = df_resid_valid[(df_resid_valid ['store_id'] == 'CA_3')&(df_resid_valid ['dept_id'] == 'HOUSEHOLD_1')]
#df_resid_train = df_resid_train[(df_resid_train ['store_id'] == 'CA_3')]
#df_resid_valid = df_resid_valid[(df_resid_valid ['store_id'] == 'CA_3')]
#df_resid_train = df_resid_train[(df_resid_train ['dept_id'] == 'FOODS_3')]
#df_resid_valid = df_resid_valid[(df_resid_valid ['dept_id'] == 'FOODS_3')]

# Train dataset
cat_feats = ['store_id', 'dept_id', 'wday', 'month', 'event_name_1', 'event_type_1',
             'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1', 'L3_event_type_1',
             'L3_event_name_1', 'event_name_2', 'event_type_2', 'snap', 'fcast_itter']

id_columns = ['date', 'day_num', 'fcast_itter_startday', 'level_fcast', 'season', 'demand_fcast', 'demand', 'resid']
train_cols = df_resid.columns[~df_resid.columns.isin(id_columns)]
X_train = df_resid_train[train_cols]
X_valid = df_resid_valid[train_cols]
X_test  = df_resid_sub[train_cols] #Change here form _sub to _test
#X_test[['resid_lag1', 'resid_lag2', 'resid_lag3', 'resid_lag4', 'resid_lag5', 'resid_lag6', 'resid_lag7', 'rmean_1_7']] = np.nan

# Label Encoding 
encod_cols = ['store_id', 'dept_id','event_name_1', 'L1_event_type_1', 'L1_event_name_1', 'L2_event_type_1', 'L2_event_name_1', 'L3_event_type_1', 'L3_event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']
for col in encod_cols:
    le = preprocessing.LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_valid[col] = le.transform(X_valid[col])
    X_test[col] = le.transform(X_test[col])
y_train = df_resid_train['resid']
y_valid  = df_resid_valid['resid']


###################
# LGBM Estimation #
###################

y_pred_val = pd.DataFrame()
y_pred_sub = pd.DataFrame()
print('LGBM Estimation Start: {}'.format(datetime.now())) 
for s in X_train['store_id'].unique():
    for d in X_train['dept_id'].unique():
        print(f'Estimation at {s} {d}')
        
        X_train_aux = X_train[(X_train['store_id'] == s) & (X_train['dept_id'] == d)]
        X_valid_aux = X_valid[(X_valid['store_id'] == s) & (X_valid['dept_id'] == d)]
        X_test_aux  = X_test[(X_test['store_id'] == s) & (X_test['dept_id'] == d)]
        y_train_aux = y_train[y_train.index.isin(X_train_aux.index)]
        y_valid_aux = y_valid[y_valid.index.isin(X_valid_aux.index)]
        
        train_data = lgb.Dataset(X_train_aux, label = y_train_aux, categorical_feature=cat_feats, free_raw_data=False)
        valid_data = lgb.Dataset(X_valid_aux, label = y_valid_aux, categorical_feature=cat_feats, free_raw_data=False)
        
        params = {
                "objective" : "regression",
                "metric" :"rmse",
        #        "force_row_wise" : True,
                "learning_rate" : 0.02,
                "early_stopping_round" : 10,
        #         "sub_feature" : 0.8,
        #        "num_leaves" : 60,
        #        "bagging_fraction" : 1,
        #        "bagging_freq" : 10,
        #        "lambda_l2" : 0.1,
        #         "nthread" : 4
            'verbosity': 0,
            'num_iterations' : 1000,
        }
         
        m_lgb = lgb.train(params, train_data, valid_sets = [valid_data], verbose_eval=100) 
        #m_lgb.save_model("model.lgb")
        
        ##########################################
        # Predict Validation sample (itterative) #
        ##########################################
                
        X_valid_oos = pd.merge(df_resid_valid[['store_id', 'dept_id', 'fcast_itter_startday', 'date', 'day_num']].rename(columns={'store_id':'store_id_aux', 'dept_id':'dept_id_aux'}),
                               X_valid_aux, how='right', left_index=True, right_index=True)
        
        #y_pred = pd.DataFrame()
        for i in range(1,6):
            X_valid_oos2 = X_valid_oos[X_valid_oos['fcast_itter_startday'] == valid_samples[f"start_sample{i}"]]
            X_valid_oos2 = X_valid_oos2[train_cols]
            #print('LGBM Forecast Start: {}'.format(datetime.now()))  
            y_pred_aux = pd.DataFrame()
            for t in range(0, len(X_valid_oos2)):
                obs = X_valid_oos2.iloc[t,:].astype('float32')
                estimate = m_lgb.predict(obs)
                y_pred_aux.loc[t,'resid_fcast'] = estimate
                '''
                X_valid_aux.loc[t+1, 'resid_lag1'] = estimate
                X_valid_aux.loc[t+1, 'resid_lag2'] = X_valid_aux.loc[t, 'resid_lag1']
                X_valid_aux.loc[t+1, 'resid_lag3'] = X_valid_aux.loc[t, 'resid_lag2']
                X_valid_aux.loc[t+1, 'resid_lag4'] = X_valid_aux.loc[t, 'resid_lag3']
                X_valid_aux.loc[t+1, 'resid_lag5'] = X_valid_aux.loc[t, 'resid_lag4']
                X_valid_aux.loc[t+1, 'resid_lag6'] = X_valid_aux.loc[t, 'resid_lag5']
                X_valid_aux.loc[t+1, 'resid_lag7'] = X_valid_aux.loc[t, 'resid_lag6']
                X_valid_aux.loc[t+1, 'rmean_1_7']  = X_valid_aux.loc[:t+1,'resid_lag1'].rolling(7).mean()[t+1]'''
            y_pred_val = y_pred_val.append(y_pred_aux)
            #print('LGBM Forecast End: {}'.format(datetime.now()))
            
        ##########################################
        # Predict Submission sample (itterative) #
        ##########################################
                    
        #y_pred = pd.DataFrame()
        #print('LGBM Forecast Start: {}'.format(datetime.now()))  
        y_pred_aux = pd.DataFrame()
        for t in range(0, len(X_test_aux)):
            obs = X_test_aux.iloc[t,:].astype('float32')
            estimate = m_lgb.predict(obs)
            y_pred_aux.loc[t,'resid_fcast'] = estimate
            '''
            X_sub_aux.loc[t+1, 'resid_lag1'] = estimate
            X_sub_aux.loc[t+1, 'resid_lag2'] = X_sub_aux.loc[t, 'resid_lag1']
            X_sub_aux.loc[t+1, 'resid_lag3'] = X_sub_aux.loc[t, 'resid_lag2']
            X_sub_aux.loc[t+1, 'resid_lag4'] = X_sub_aux.loc[t, 'resid_lag3']
            X_sub_aux.loc[t+1, 'resid_lag5'] = X_sub_aux.loc[t, 'resid_lag4']
            X_sub_aux.loc[t+1, 'resid_lag6'] = X_sub_aux.loc[t, 'resid_lag5']
            X_sub_aux.loc[t+1, 'resid_lag7'] = X_sub_aux.loc[t, 'resid_lag6']
            X_sub_aux.loc[t+1, 'rmean_1_7']  = X_sub_aux.loc[:t+1,'resid_lag1'].rolling(7).mean()[t+1]'''
        y_pred_sub = y_pred_sub.append(y_pred_aux)
        #print('LGBM Forecast End: {}'.format(datetime.now()))
print('LGBM Forecast End: {}'.format(datetime.now()))  

#############################
# Calculate validation RMSE #
#############################
start_sample_list = [] 
for i in range(1,6):
    start_sample_list.append(valid_samples[f"start_sample{i}"])   
    
RMSE_valid = df_resid_valid[['store_id', 'dept_id', 'fcast_itter_startday', 'date', 'day_num', 'demand', 'demand_fcast']]
RMSE_valid = RMSE_valid[RMSE_valid['fcast_itter_startday'].isin(start_sample_list)].reset_index(drop=True)
RMSE_valid = RMSE_valid.sort_values(by=['fcast_itter_startday', 'store_id', 'dept_id'])
RMSE_valid['resid_fcast'] = y_pred_val.reset_index(drop=True)
RMSE_valid['demand_fcast_final'] = RMSE_valid['demand_fcast'] + RMSE_valid['resid_fcast']

from sklearn.metrics import mean_squared_error
from math import sqrt
for i in RMSE_valid['fcast_itter_startday'].unique():    
    VARX_RMSE = sqrt(mean_squared_error(RMSE_valid[RMSE_valid['fcast_itter_startday']==i].demand, RMSE_valid[RMSE_valid['fcast_itter_startday']==i].demand_fcast))
    GBM_RMSE  = sqrt(mean_squared_error(RMSE_valid[RMSE_valid['fcast_itter_startday']==i].demand, RMSE_valid[RMSE_valid['fcast_itter_startday']==i].demand_fcast_final))
    print(f'{i}: VARX/GBM RMSE: {VARX_RMSE} / {GBM_RMSE}')
       
##########################################################################
# Calculate final demand prediction (level_fcast + season + resid_fcast) #
##########################################################################        
demand_fcast_final = df_resid_sub.merge(pd.DataFrame(y_pred_sub).reset_index(drop=True), how='left', left_index=True, right_index=True)
demand_fcast_final['demand_fcast_final'] = demand_fcast_final['demand_fcast'] + demand_fcast_final['resid_fcast']

###############
#  Submission #
###############

#Calculate weights
def read_data():
    print('Reading files...')
    calendar = pd.read_csv(f'{path}calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(f'{path}sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(f'{path}sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(f'{path}sample_submission.csv')
    return calendar, sell_prices, sales_train_validation, submission

cal_df, price_df, df, sub = read_data()

cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))
price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"

for day in tqdm(range(1885, 1914)): # Change here for second submission
    wk_id = list(cal_df[cal_df["d"]==day]["wm_yr_wk"])[0]
    wk_price_df = price_df[price_df["wm_yr_wk"]==wk_id]
    df = df.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')
    df["unit_sales_" + str(day)] = df["sell_price"] * df["d_" + str(day)]
    df.drop(columns=["sell_price"], inplace=True)
    
df["dollar_sales"] = df[[c for c in df.columns if c.find("unit_sales")==0]].sum(axis=1)
df.drop(columns=[c for c in df.columns if c.find("unit_sales")==0], inplace=True)
grouped_dollar_sales = df.groupby(['store_id', 'dept_id'])["dollar_sales"].sum().reset_index().rename(columns = {"dollar_sales":"grouped_dollar_sales"})
df = df.merge(grouped_dollar_sales, how='left', left_on=['store_id', 'dept_id'], right_on=['store_id', 'dept_id'])
df["weight"] = df["dollar_sales"] / df["grouped_dollar_sales"]
df = df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'weight']]

print(df.groupby(['store_id', 'dept_id']).sum())

#Prepare submission file
sub = pd.DataFrame(sub['id'])
aux = sub['id'].str.split("_", 6, expand=True)
sub['item_id'] = aux[0] + "_" + aux[1] + "_" + aux[2]
sub['dept_id'] = aux[0] + "_" + aux[1]
sub['store_id'] = aux[3] + "_" + aux[4]
sub['state_id'] = aux[3]

# Add weights to submission
sub = sub.merge(df[['item_id', 'store_id', 'weight']], how='left', left_on=['item_id', 'store_id'], right_on=['item_id', 'store_id'])

# Add forecasts to submission
sub = sub.merge(demand_fcast_final[['store_id', 'dept_id', 'day_num', 'demand_fcast_final']], how='left', left_on=['store_id', 'dept_id'], right_on=['store_id', 'dept_id'])
sub['demand_submission'] = sub['demand_fcast_final'] * sub['weight']

# Transpose
sub = sub.pivot(index='id', columns='day_num', values='demand_submission').reset_index()
sub.columns = submission.columns

path_results = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy/"
sub.to_csv(f'{path_results}submission.csv', index=False)

#sub2 = sub.copy()
#sub2.iloc[:,1:] = sub2.iloc[:,1:] * 1.05
#path_results = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy/"
#sub2.to_csv(f'{path_results}submission_2.csv', index=False)

######################
#  Feature Imprtance #
######################
feature_imp = pd.DataFrame(sorted(zip(m_lgb.feature_importance(),X_train.columns)), columns=['Value','Feature'])

##################
#  Day num count #
##################
count_day = df_resid.groupby('day_num')['day_num'].agg('count')
