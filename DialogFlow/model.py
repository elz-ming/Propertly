import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

import pickle

SALE_URL = 'https://raw.githubusercontent.com/elz-ming/Propertly/main/data/iProperty_sale_to_be_trained.csv'
RENT_URL = 'https://raw.githubusercontent.com/elz-ming/Propertly/main/data/iProperty_rent_to_be_trained.csv'

#========== Handing SALE Data ==========#
sale_df = pd.read_csv(SALE_URL)

sale_df.drop(['name','psf','area'], axis=1, inplace=True)

sale_cat_col = sale_df.select_dtypes([object]).columns.tolist()
sale_num_col = sale_df.select_dtypes([int,float]).drop(['price'], axis=1).columns.tolist()
sale_tar_col = sale_df[['price']].columns.tolist()

for col in sale_tar_col:
    q_low = sale_df[col].quantile(0.01)
    q_hi  = sale_df[col].quantile(0.99)
    sale_df = sale_df[(sale_df[col] < q_hi) & (sale_df[col] > q_low)]

sale_ohe = OneHotEncoder()
sale_ohe.fit(sale_df[sale_cat_col])
sale_cat_arr = sale_ohe.transform(sale_df[sale_cat_col]).toarray()

sale_num_arr = sale_df.drop(sale_cat_col + sale_tar_col, axis=1).to_numpy()

sale_tar_arr = sale_df.price.to_numpy()

sale_x = np.concatenate((sale_num_arr,sale_cat_arr),axis=1)
sale_y = sale_tar_arr
sale_x_train, sale_x_test, sale_y_train, sale_y_test = train_test_split(sale_x,sale_y,test_size=0.2)

with open(r'sale_ohe.pkl','wb') as sale_ohe_pkl:
    pickle.dump(sale_ohe, sale_ohe_pkl, protocol=2)

#========== Handing RENT Data ==========#
rent_df = pd.read_csv(RENT_URL)

rent_df.drop(['name','psf','area'], axis=1, inplace=True)

rent_cat_col = rent_df.select_dtypes([object]).columns.tolist()
rent_num_col = rent_df.select_dtypes([int,float]).drop(['price'], axis=1).columns.tolist()
rent_tar_col = rent_df[['price']].columns.tolist()

for col in rent_tar_col:
    q_low = rent_df[col].quantile(0.01)
    q_hi  = rent_df[col].quantile(0.99)
    rent_df = rent_df[(rent_df[col] < q_hi) & (rent_df[col] > q_low)]


rent_ohe = OneHotEncoder()
rent_ohe.fit(rent_df[rent_cat_col])
rent_cat_arr = rent_ohe.transform(rent_df[rent_cat_col]).toarray()

rent_num_arr = rent_df.drop(rent_cat_col + rent_tar_col, axis=1).to_numpy()

rent_tar_arr = rent_df.price.to_numpy()

rent_x = np.concatenate((rent_num_arr,rent_cat_arr),axis=1)
rent_y = rent_tar_arr
rent_x_train, rent_x_test, rent_y_train, rent_y_test = train_test_split(rent_x,rent_y,test_size=0.2)

with open(r'rent_ohe.pkl','wb') as rent_ohe_pkl:
    pickle.dump(rent_ohe, rent_ohe_pkl, protocol=2)

#========== Creating MODEL ==========#
PARAMETERS = {
    'ccp_alpha': 0.1,
    'loss': 'squared_error',
    'max_depth': 10,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 100,
    'validation_fraction': 0.25,
}

sale_model = GradientBoostingRegressor()
sale_model.set_params(**PARAMETERS)
sale_model.fit(sale_x_train,sale_y_train)
with open(r'sale_model.pkl','wb') as sale_model_pkl:
    pickle.dump(sale_model, sale_model_pkl, protocol=2)

rent_model = GradientBoostingRegressor()
rent_model.set_params(**    PARAMETERS)
rent_model.fit(rent_x_train,rent_y_train)
with open(r'rent_model.pkl','wb') as rent_model_pkl:
    pickle.dump(rent_model, rent_model_pkl, protocol=2)

