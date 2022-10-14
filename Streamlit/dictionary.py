from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

#========== CODE BELOW ==========#
#========== TO BE IMPORTED ==========#
model_dict = {
    'dt' : 'Decision Tree Regressor',
    'rf' : 'Random Forest Regressor',
    'sv' : 'Support Vector Machine Regressor',
    'kn' : 'K Neighbors Regressor',
    'gb' : 'Gradient Boosting Regressor',
    'nn' : 'Neural Network Regressor',
    'xg' : 'XGB Regressor',
}

SEED = 42
pipeline_dict = {
    'dt': DecisionTreeRegressor(random_state=SEED),
    'rf': RandomForestRegressor(random_state=SEED),
    'sv': SVR(),
    'kn': KNeighborsRegressor(),
    'gb': GradientBoostingRegressor(random_state=SEED),
    'nn': MLPRegressor(random_state=SEED),
    'xg': XGBRegressor(seed=SEED),
}

hyperparameters_dict = {
    'dt' : {},
    'rf' : {},
    'sv' : {},
    'kn' : {},
    'gb' : {},
    'nn' : {},
    'xg' : {},
}