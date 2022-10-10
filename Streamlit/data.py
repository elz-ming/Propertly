from Streamlit.functions import feature_engineering

import pandas as pd

from sklearn.model_selection import train_test_split

SALE_DATA_URL = ("https://raw.githubusercontent.com/elz-ming/Propertly/main/data/iProperty_sale_to_be_trained.csv")
RENT_DATA_URL = ("https://raw.githubusercontent.com/elz-ming/Propertly/main/data/iProperty_rent_to_be_trained.csv")

sale_data = pd.read_csv(SALE_DATA_URL)
sale_data_fe = feature_engineering(sale_data)
sale_x = sale_data_fe.drop(['price'],axis=1)
sale_y = sale_data_fe['price']

rent_data = pd.read_csv(RENT_DATA_URL)
rent_data_fe = feature_engineering(rent_data)
rent_x = rent_data_fe.drop(['price'],axis=1)
rent_y = rent_data_fe['price']

#========== TO BE IMPORTED ==========#
sale_x_train, sale_x_test, sale_y_train, sale_y_test = train_test_split(sale_x, sale_y, test_size=0.2)
rent_x_train, rent_x_test, rent_y_train, rent_y_test = train_test_split(rent_x, rent_y, test_size=0.2)