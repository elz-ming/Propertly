import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

st.title("PROPERTLY")
st.header("Handle Property, Properly")
st.markdown("This is a simple app to demonstrate the performance of supervised machine learning models (regression) on datasets scaped from the internet.")
st.markdown("The dataset is")

st.sidebar.title('Model')

regressor_input = st.sidebar.selectbox(
    'Regressor',
    ('Linear Regressor', 'Decision Tree Regressor', 'Random Forest Regressor', 'Support Vector Machine Regressor',  'K Neighbors Regressor', 'Gradient Boosting Regressor', 'Neural Network Regressor')
)

model_dict = {
    'lr' : 'Linear Regressor',
    'dt' : 'Decision Tree Regressor',
    'rf' : 'Random Forest Regressor',
    'svm': 'Support Vector Machine Regressor',
    'kn' : 'K Neigbors Regressor',
    'gb' : 'Gradient Boosting Regressor',
    'nn' : 'Neural Network Regressor',
}

SEED = 42
pipeline_dict = {
    'lr' : make_pipeline(StandardScaler(), LinearRegression()),
    'dt' : make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=SEED)),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor(random_state=SEED)),
    'svm': make_pipeline(StandardScaler(), SVR()),
    'kn' : make_pipeline(StandardScaler(), KNeighborsRegressor()),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=SEED)),
    'nn' : make_pipeline(StandardScaler(), MLPRegressor(random_state=SEED)),
}

hyperparameters_dict = {
    'lr' : {},
    'dt' : {},
    'rf' : {},
    'svm': {},
    'kn' : {},
    'gb' : {},
    'nn' : {},
}

for key,value in model_dict.items():
    if value == regressor_input:
        model_id = key

model = GridSearchCV(pipeline_dict[model_id], hyperparameters_dict[model_id],verbose=4)



def regressor(Model):
    for key,value in model_dict.items():
        if value == Model:
            model_id = key
    return None