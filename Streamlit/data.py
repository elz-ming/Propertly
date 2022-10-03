from functions import feature_engineering

import pandas as pd

from sklearn.model_selection import train_test_split

SALE_DATA_URL = ("https://raw.githubusercontent.com/elz-ming/Propertly/main/JupyterNotebook/data/iProperty_sale_to_be_trained.csv")
RENT_DATA_URL = ("")

sale_data = pd.read_csv(SALE_DATA_URL)
sale_data = feature_engineering(sale_data)
sale_x = sale_data.drop(['price'],axis=1)
sale_y = sale_data['price']

rent_data = None
rent_x = None
rent_y = None

#========== TO BE IMPORTED ==========#
sale_x_train, sale_x_test, sale_y_train, sale_y_test = train_test_split(sale_x, sale_y, test_size=0.2)
# rent_x_train, rent_x_test, rent_y_train, rent_y_test = train_test_split(rent_x, rent_y, test_size=0.2)