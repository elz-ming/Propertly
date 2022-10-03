from cgi import test
from dictionary import *
from data import sale_data, rent_data
from data import sale_x_train, sale_x_test, sale_y_train, sale_y_test
from data import rent_x_train, rent_x_test, rent_y_train, rent_y_test

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import GridSearchCV

from functions import regressor

#========== CODE BELOW ==========#

#Main Body
st.title("PROPERTLY")
st.header('Introduction')
st.markdown("This is a simple app to demonstrate the performance of supervised machine learning models (regression) on two datasets scaped from the internet.")
st.markdown("The datasets consists of : ")

col1, col2 = st.columns(2)

with col1:
    st.markdown("1. Property selling price listed on iProperty, features : District, State, Type, Details, Bedrooms, Bathrooms, Carparks")
    show_sale = st.checkbox('Show SALE DataFrame')
    if show_sale:
        st.markdown('Rows : {}, Columns : {}'.format(sale_data.shape[0],sale_data.shape[1]))
        sale_data
with col2:
    st.markdown("2. Property rental price listed on iProperty, features : District, State, Type, Details, Bedrooms, Bathrooms, Carparks")
    show_rent = st.checkbox('Show RENT DataFrame')
    if show_rent:
        st.markdown('Rows : {}, Columns : {}'.format(rent_data.shape[0],rent_data.shape[1]))
        rent_data

#Side Bar
st.sidebar.title("Let's Build Model(s)")

type_input = st.sidebar.radio(
    label="Method",
    label_visibility="collapsed",
    options=('Single Model','Multi Models')
)

if type_input == 'Single Model':
    regressor_input = st.sidebar.selectbox(
        'Regressor',
        ('Decision Tree Regressor', 'Random Forest Regressor', 'Support Vector Machine Regressor',  'K Neighbors Regressor', 'Gradient Boosting Regressor', 'Neural Network Regressor')
    )
    st.sidebar.subheader('Parameters')

    if regressor_input == 'Decision Tree Regressor':
        model_id = 'dt'
        hyperparameters = {
            'decisiontreeregressor__min_samples_leaf'         : [st.sidebar.number_input(label='min_samples_leaf', min_value=1, value=1)],
            'decisiontreeregressor__min_samples_split'        : [st.sidebar.number_input(label='min_samples_split', min_value=2, value=2)],
            'decisiontreeregressor__max_depth'                : [st.sidebar.number_input(label='max_depth', min_value=1, value=5)],
            'decisiontreeregressor__min_weight_fraction_leaf' : [st.sidebar.number_input(label='min_weight_fraction_leaf', min_value=0.0, value=0.0)]
        }

    elif regressor_input == 'Random Forest Regressor':
        model_id = 'rf'
        hyperparameters = {
            'randomforestregressor__n_estimators'             : [st.sidebar.number_input(label='n_estimators', min_value=10, value=100)],            
            'randomforestregressor__min_samples_leaf'         : [st.sidebar.number_input(label='min_samples_leaf', min_value=1, value=1)],
            'randomforestregressor__min_samples_split'        : [st.sidebar.number_input(label='min_samples_split', min_value=2, value=2)],
            'randomforestregressor__max_depth'                : [st.sidebar.number_input(label='max_depth', min_value=1, value=5)],
            'randomforestregressor__min_weight_fraction_leaf' : [st.sidebar.number_input(label='min_weight_fraction_leaf', min_value=0.0, value=0.0)],
        }

    elif regressor_input == 'Support Vector Machine Regressor':
        model_id = 'sv'
        hyperparameters = {
            'svr__kernel'  : [st.sidebar.selectbox(label='kernel', options=('poly', 'rbf', 'sigmoid'), index=1)],
            'svr__C'       : [st.sidebar.number_input(label='C', min_value=1.0, value=1.0)],
            'svr__gamma'   : [st.sidebar.selectbox(label='gamma', options=('scale', 'auto'), index=0)],
            'svr__epsilon' : [st.sidebar.number_input(label='epsilon', min_value=0.1, value=0.1)],
        }

    elif regressor_input == 'K Neighbors Regressor':
        model_id = 'kn'
        hyperparameters = {
            'kneighborsregressor__algorithm' : [st.sidebar.selectbox(label='algorithm', options=('auto', 'ball_tree', 'kd_tree'), index=0)],
            'kneighborsregressor__leaf_size' : [st.sidebar.number_input(label='leaf_size', min_value=10, value=30)],
            'kneighborsregressor__weights'   : [st.sidebar.selectbox(label='weights', options=('uniform', 'distance'), index = 0)],
        }

    elif regressor_input == 'Gradient Boosting Regressor':
        model_id = 'gb'
        hyperparameters = {
            'gradientboostingregressor__learning_rate'       : [st.sidebar.number_input(label='learning rate', min_value=0.001, value=0.1)],
            'gradientboostingregressor__n_estimators'        : [st.sidebar.number_input(label='n_estimators', min_value=10, value=100)],
            'gradientboostingregressor__validation_fraction' : [st.sidebar.number_input(label='validation_fraction', min_value=0.01, value=0.1)],
            'gradientboostingregressor__ccp_alpha'           : [st.sidebar.number_input(label='ccp_alpha', min_value=0.0, value=0.0)],
            'gradientboostingregressor__min_samples_leaf'    : [st.sidebar.number_input(label='min_samples_leaf', min_value=1, value=1)],
            'gradientboostingregressor__min_samples_split'   : [st.sidebar.number_input(label='min_samples_split', min_value=2, value=2)],
            'gradientboostingregressor__max_depth'           : [st.sidebar.number_input(label='max_depth', min_value=1, value=3)]
        }

    elif regressor_input == 'Neural Network Regressor':
        model_id = 'nn'
        first = st.sidebar.number_input(label='first layer', min_value=1, max_value=1000, value = 100)
        second = st.sidebar.number_input(label='second layer', min_value=0, max_value=1000, value = 0)
        third = st.sidebar.number_input(label='third layer', min_value=0, max_value=1000, value = 0)
        if second == 0 & third == 0:
            hidden_layer_sizes = (first,)
        elif third == 0:
            hidden_layer_sizes = (first, second,)
        else:
            hidden_layer_sizes = (first, second, third,)
        hyperparameters = {
            'mlpregressor__activation'          : [st.sidebar.selectbox(label='activation', options=('identity', 'logistic', 'tanh', 'relu'), index=3)],
            'mlpregressor__solver'              : [st.sidebar.selectbox(label='activation', options=('lbfgs', 'sgd', 'adam'), index=2)],
            'mlpregressor__alpha'               : [st.sidebar.number_input(label='alpha', min_value=0.0001, value=0.0001)],
            'mlpregressor__learning_rate'       : [st.sidebar.selectbox(label='learning rate', options=('constant', 'invscaling', 'adaptive'), index=0)],
            'mlpregressor__learning_rate_init'  : [st.sidebar.number_input(label='learning_rate_init', min_value=0.001, value=0.001)],
            'mlpregressor__max_iter'            : [st.sidebar.number_input(label='epochs', min_value=1, max_value=1000, value=200)],
            'mlpregressor__validation_fraction' : [st.sidebar.number_input(label='validation_fraction', min_value=0.01, value=0.1)],
            'mlpregressor__hidden_layer_sizes'  : [hidden_layer_sizes],
        }



    if st.sidebar.button('Train'):
        sale_metrics_dict = regressor(
                pipeline_dict[model_id],
                hyperparameters,
                sale_x_train,
                sale_x_test,
                sale_y_train,
                sale_y_test
            )
        rent_metrics_dict = regressor(
                pipeline_dict[model_id],
                hyperparameters,
                rent_x_train,
                rent_x_test,
                rent_y_train,
                rent_y_test
            )

        with col1:
            st.subheader("SALE DataSet ({})".format(regressor_input))
            for key, value in sale_metrics_dict.items():
                st.markdown('{} : {}'.format(key, value))
        
        with col2:
            st.subheader("RENT DataSet ({})".format(regressor_input))
            for key, value in rent_metrics_dict.items():
                st.markdown('{} : {}'.format(key, value))


elif type_input == 'Multi Models':
    regressor_input = st.sidebar.multiselect(
        'Regressors',
        ['Decision Tree Regressor', 'Random Forest Regressor', 'Support Vector Machine Regressor',  'K Neighbors Regressor', 'Gradient Boosting Regressor', 'Neural Network Regressor']
    )

    selected_model_ids = []
    for key,value in model_dict.items():
        if any(value in t for t in regressor_input):
            selected_model_ids.append(key)

    model_id_list =[]
    model_list =[]

    train_score_list_sale = []
    test_score_list_sale  = []
    R2_list_sale          = []
    RMSE_list_sale        = []
    MAE_list_sale         = []
    MAPE_list_sale        = []

    train_score_list_rent = []
    test_score_list_rent  = []
    R2_list_rent          = []
    RMSE_list_rent        = []
    MAE_list_rent         = []
    MAPE_list_rent        = []

    for model_id in selected_model_ids:
        sale_metrics_dict = regressor(
            pipeline_dict[model_id],
            hyperparameters_dict[model_id],
            sale_x_train,
            sale_x_test,
            sale_y_train,
            sale_y_test,
        )

        rent_metrics_dict = regressor(
            pipeline_dict[model_id],
            hyperparameters_dict[model_id],
            rent_x_train,
            rent_x_test,
            rent_y_train,
            rent_y_test,
        )

        model_id_list.append(model_id)
        model_list.append(model_dict[model_id])

        train_score_list_sale.append(sale_metrics_dict['train_score'])
        test_score_list_sale.append(sale_metrics_dict['test_score'])
        R2_list_sale.append(sale_metrics_dict['R2'])
        RMSE_list_sale.append(sale_metrics_dict['RMSE'])
        MAE_list_sale.append(sale_metrics_dict['MAE'])
        MAPE_list_sale.append(sale_metrics_dict['MAPE'])

        train_score_list_rent.append(rent_metrics_dict['train_score'])
        test_score_list_rent.append(rent_metrics_dict['test_score'])
        R2_list_rent.append(rent_metrics_dict['R2'])
        RMSE_list_rent.append(rent_metrics_dict['RMSE'])
        MAE_list_rent.append(rent_metrics_dict['MAE'])
        MAPE_list_rent.append(rent_metrics_dict['MAPE'])


    model2sale_metrics_dict = {
        'model_id'    : model_id_list,
        'model'       : model_list,
        'train_score' : train_score_list_sale,
        'test_score'  : test_score_list_sale,
        'R2'          : R2_list_sale,
        'RMSE'        : RMSE_list_sale,
        'MAE'         : MAE_list_sale,
        'MAPE'        : MAPE_list_sale,
    }

    model2rent_metrics_dict = {
        'model_id'    : model_id_list,
        'model'       : model_list,
        'train_score' : train_score_list_rent,
        'test_score'  : test_score_list_rent,
        'R2'          : R2_list_rent,
        'RMSE'        : RMSE_list_rent,
        'MAE'         : MAE_list_rent,
        'MAPE'        : MAPE_list_rent,
    }

    sale_metrics_df = pd.DataFrame.from_dict(model2sale_metrics_dict)
    rent_metrics_df = pd.DataFrame.from_dict(model2rent_metrics_dict)

    st.sidebar.subheader('Metrics')
    train_score = st.sidebar.checkbox('Train Score')
    test_score = st.sidebar.checkbox('Test Score')
    R2 = st.sidebar.checkbox('R2')
    RMSE = st.sidebar.checkbox('RMSE')
    MAE = st.sidebar.checkbox('MAE')
    MAPE = st.sidebar.checkbox('MAPE')

    if train_score:
        col1.bar_chart(data=sale_metrics_df, x='model', y='train_score')
        col2.bar_chart(data=rent_metrics_df, x='model', y='train_score')
    if test_score:
        col1.bar_chart(data=sale_metrics_df, x='model', y='test_score')
        col2.bar_chart(data=rent_metrics_df, x='model', y='test_score')
    if R2:
        col1.bar_chart(data=sale_metrics_df, x='model', y='R2')
        col2.bar_chart(data=rent_metrics_df, x='model', y='R2')
    if RMSE:
        col1.bar_chart(data=sale_metrics_df, x='model', y='RMSE')
        col2.bar_chart(data=rent_metrics_df, x='model', y='RMSE')
    if MAE:
        col1.bar_chart(data=sale_metrics_df, x='model', y='MAE')
        col2.bar_chart(data=rent_metrics_df, x='model', y='MAE')
    if MAPE:
        col1.bar_chart(data=sale_metrics_df, x='model', y='MAPE')
        col2.bar_chart(data=rent_metrics_df, x='model', y='MAPE')