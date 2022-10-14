import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

#========== CODE BELOW ==========#
#========== TO BE IMPORTED ==========#
def feature_engineering(DataFrame):

    df = DataFrame.copy()

    #1 Dropping redundant columns
    df.drop(['name','psf','area'], axis=1, inplace=True)

    #2 Removing outliers in target
    for col in ['price']:
        q_low = df[col].quantile(0.01)
        q_hi  = df[col].quantile(0.99)
        df = df[(df[col] < q_hi) & (df[col] > q_low)]

    #3 OneHotEncoding categorical features
    one_hot = pd.get_dummies(df[['district','state','type','details']])
    df = df.join(one_hot).drop(['district','state','type','details'], axis=1)

    return df

def regressor(Model, Hyperparameters, x_train, x_test, y_train, y_test):
    model = Model
    model.set_params(**Hyperparameters)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics_dict = {
        'R2'    : r2_score(y_test, y_pred),
        'RMSE'  : mean_squared_error(y_test, y_pred, squared=False),
        'MAE'   : mean_absolute_error(y_test, y_pred),
        'MAPE'  : mean_absolute_percentage_error(y_test, y_pred),
    }
    return metrics_dict