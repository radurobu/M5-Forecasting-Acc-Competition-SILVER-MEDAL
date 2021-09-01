# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:34:38 2020

@author: Radu
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
#import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime, timedelta
#import lightgbm as lgb
#import dask_xgboost as xgb
#import dask.dataframe as dd
from sklearn import preprocessing
import lightgbm as lgb
import pickle
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
path = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy\Data Evaluation/"
path_results = r"C:\Users\Radu\Desktop\ML Projects\M5 Accuracy/"


CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

pd.options.display.max_columns = 50

h = 28 
max_lags = 57
tr_last = 1941
fday = datetime(2016,5, 23) 
fday

def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv(f"{path}sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv(f"{path}calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(f"{path}sales_train_evaluation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
            
FIRST_DAY = 500 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !

df = create_dt(is_train=True, first_day= FIRST_DAY)
df.shape

create_fea(df)
df.shape

cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]

np.random.seed(64)

fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 
                         categorical_feature=cat_feats, free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                              categorical_feature=cat_feats,
                 free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!

del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()

params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1200,
    'num_leaves': 2**11-1,
    "min_data_in_leaf":  2**12-1,
}

m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20)

# save the model to disk
pickle.dump(m_lgb, open(f"{path_results}model_under05_sub.lgb", 'wb'))
 
# load the model from disk
#m_lgb = pickle.load(open(f"{path_results}model_under05.lgb", 'rb'))


def create_lag_features_for_test(dt, day):
    # create lag feaures just for single day (faster)
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt.date == day, lag_col] = \
            dt.loc[dt.date ==day-timedelta(days=lag), 'sales'].values  # !!! main

    windows = [7, 28]
    for window in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt.date==day,'id'])
            dt.loc[dt.date == day,f"rmean_{lag}_{window}"] = \
                df_window_grouped.sales.values
                

def create_date_features_for_test(dt):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")



te0 = create_dt(False)  # create master copy of `te`
create_date_features_for_test (te0)

te = te0.copy()  # just copy
cols = [f"F{i}" for i in range(1, 29)]

for tdelta in range(0, 28):
    day = fday + timedelta(days=tdelta)
    print(tdelta, day.date())
    tst = te[(te.date >= day - timedelta(days=max_lags))
             & (te.date <= day)].copy()
#         create_fea(tst)  # correct, but takes much time
    create_lag_features_for_test(tst, day)  # faster  
    tst = tst.loc[tst.date == day, train_cols]
    te.loc[te.date == day, "sales"] =  m_lgb.predict(tst)  # magic multiplier by kyakovlev

te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
te_sub.fillna(0., inplace=True)
te_sub.sort_values("id", inplace=True)
te_sub.reset_index(drop=True, inplace=True)

te_sub2 = te_sub.copy()
te_sub2['id'] = te_sub2['id'].map(lambda x: x.replace("evaluation","validation"))

te_sub_final = te_sub.append(te_sub2).reset_index(drop=True)
te_sub_final.to_csv(f"{path_results}submission_final.csv", index=False)









