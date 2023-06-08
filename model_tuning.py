from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, make_scorer, precision_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline

import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier

from keras import models, layers
from keras.metrics import Precision


import pandas as pd
import numpy as np
import time
import math

import settings
from utils import get_x_y_w


def get_neuralnetwork_classifier(n_features):
    tf.random.set_seed(123)

    neurons = np.logspace(3, 10, num=8, base=2, dtype=int)
    dropout_rates = np.linspace(0.1, 0.8, num=8)
    learning_rates = np.logspace(-4, 1, 6)
    batch_size = np.logspace(4, 12, num=9, base=2, dtype=int)
    epochs = [5, 10, 50, 100]

    param_grid = dict(batch_size=batch_size, epochs=epochs, model__dropout_rate=dropout_rates, optimizer__learning_rate=learning_rates, model__neurons=neurons)
    #print(param_grid)

    def create_model(dropout_rate, neurons):
        model = models.Sequential()
        model.add(Dense(units=neurons, activation='relu', input_shape=(n_features,)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    model = KerasClassifier(model=create_model, verbose=0)
    return (model, param_grid)



def hypertune_model(X, y, scoring_list, clf_name, search):
    results = []
    variables = ['scoring', 'best_score', 'best_params', 'running_time']
    for scoring in scoring_list:
        print(f"""\n************************************************************************************
            \nHypertuning {clf_name} for {scoring}\n\n************************************************************************************
            """)
        search.scoring = scoring
        start = time.time()
        search.fit(X, y)
        end = time.time()
        running_time = math.trunc(time.time() - start)
        results.append([scoring, search.best_score_, search.best_params_, running_time])
        print(f'Best parameters (score={search.best_score_})')
        print(search.best_params_)
        print(f'Running time: {math.trunc(end-start)} seconds')
    results_df = pd.DataFrame(data=results, columns=variables)
    csv_name = f'{settings.path_hypertuning}/{clf_name}.csv'
    results_df.to_csv(csv_name, index=False)
    print('Saved results to', csv_name)

def hypertune_model_weighted(X, y, scoring_list, clf_name, search, sample_weight):    
    results = []
    variables = ['scoring', 'best_score', 'best_params', 'running_time']
    for scoring in scoring_list:
        print(f"""\n************************************************************************************
            \nHypertuning {clf_name} for {scoring}\n\n************************************************************************************
            """)
        search.scoring = scoring
        start = time.time()
        search.fit(X, y, sample_weight=sample_weight)
        end = time.time()
        running_time = math.trunc(time.time() - start)
        results.append([scoring, search.best_score_, search.best_params_, running_time])
        print(f'Best parameters (score={search.best_score_})')
        print(search.best_params_)
        print(f'Running time: {math.trunc(end-start)} seconds')
    results_df = pd.DataFrame(data=results, columns=variables)
    csv_name = f'{settings.path_hypertuning}/{clf_name}.csv'
    results_df.to_csv(csv_name, index=False)
    print('Saved results to', csv_name)

#model_tuning()
#best_gbt()
#best_adaboost()
#randomforest_tuning()
#adaboost_tuning()
#xgboost_tuning()


def model_tuning_():
    # split data in training and test set
    # in this case, the training set is used for hyperparameter tuning with cross validation
    # the test set is used with the best scoring models for more accurate metric evaluation and for cost evaluation
    df = pd.read_csv(settings.data_features_csv)
    y = df.pop('winner')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, stratify=y, random_state=123)
    odds_train = X_train.pop('p0_odds')
    X_train.pop('p1_odds')
    odds_test = X_test.pop('p0_odds')
    #logisticregression_tuning(X_train, y_train, odds_train)
    #randomforest_tuning(X_train, y_train, odds_train)
    neuralnetwork_tuning(X_train, y_train, odds_train)


def compare_features_hypertune(): 
    param_grid =[
        {  
            "n_estimators" : [1000],
            "max_features": ['sqrt', 'log2', None],
            "min_samples_split": [2, 5, 10],
            "criterion" : ['gini', 'entropy']
        }
    ]
    clf = RandomForestClassifier(random_state=123)  
    df1 = pd.read_csv(settings.data_features_csv)
    df1.drop(columns=['p0_odds', 'p1_odds'], inplace=True)
    df2 = pd.read_csv(settings.data_features2_csv)

    results = []
    variables = ['best_score', 'best_params', 'running_time']
    for df in [df1, df2]:
        print('Scoring features...')
        y = df.pop('winner')
        X = df
        search = GridSearchCV(clf, param_grid, n_jobs=-1, scoring='accuracy')
        start = time.time()
        search.fit(X, y)
        end = time.time()
        running_time = math.trunc(end - start)
        results.append([search.best_score_, search.best_params_, running_time])
        print(f'Best parameters: {search.best_score_}')
        print(search.best_params_, '\n\n')
    results_df = pd.DataFrame(data=results, columns=variables)
    csv_name = f'{settings.path_hypertuning}/feature_comparison.csv'
    results_df.to_csv(csv_name, index=False)
    print('Saved results to', csv_name)

def model_tuning():

    #scoring_list = ['precision', 'f1', 'accuracy']
    scoring_list = ['accuracy']

    # split data in training and test set
    # in this case, the training set is used for hyperparameter tuning with cross validation
    # the test set is used with the best scoring models for more accurate metric evaluation and for cost evaluation
    (X, y, w) = get_x_y_w(settings.data_features_csv)
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=123))
    lr_grid = [
        {
            'logisticregression__penalty' : ['l2', None],
            'logisticregression__C' : np.logspace(-10, 9, 20),
            'logisticregression__solver' : ['lbfgs','newton-cholesky','saga'],
        }
    ]
    lr_search = GridSearchCV(lr, lr_grid, n_jobs=-2)

    rf =   RandomForestClassifier(random_state=123)  
    rf_grid = {
        "n_estimators" : [10, 100],
        "max_depth": [None, 3, 10],
        "max_features": ['sqrt', 'log2', None],
        "min_samples_split": [2, 5,10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
    rf_search = RandomizedSearchCV(rf, rf_grid, n_jobs=-2, n_iter=30)

    xgb = XGBClassifier(random_state=123)
    xgb_grid =  {
        'learning_rate':[0.3, 0.1, 0.6, 1],
        'n_estimators': [100, 200],
        'max_depth' : [6, 2, 12, 24],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1, 10],
        'reg_alpha' : [0, 0.1, 1, 2] 
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_grid, n_jobs=-2, n_iter=30)

    (nn, nn_grid) = get_neuralnetwork_classifier(len(X.columns))
    nn_search = RandomizedSearchCV(nn, nn_grid, n_jobs=-2, n_iter=20)

    clf_list = [
        #(lr_search, 'logistic_regression'),
        #(rf_search, 'random_forest'),
        (xgb_search, 'xgboost'),
        (nn_search, 'neural_network')
    ]

    for (clf_search, clf_name) in clf_list:
        print('Tuning', clf_name)
        hypertune_model_weighted(X=X, y=y, scoring_list=scoring_list, search=clf_search, clf_name=clf_name, sample_weight=w)



def xgboost_tuning():

    scoring = make_scorer(fbeta_score, beta=0.05)
    #scoring = make_scorer(fbeta_score, beta=0.1, pos_label=0)

    print('xgboost_tuning')
    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 6],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.1, 0.01],
        'reg_lambda': [0.1, 0.01],
        'min_child_weight': [1, 5]
    }

    # Create an XGBoost classifier
    xgb = XGBClassifier(random_state=123)

    df = pd.read_csv(settings.data_features_csv)
    df.drop(columns=['p0_odds', 'p1_odds'], inplace=True)
    y_train, X_train = df.pop('winner'), df

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(xgb, param_grid, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(best_params, '\n', best_score)



def xgboost_tuning_():
    print('xgboost_tuning')
    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 6],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.1, 0.01],
        'reg_lambda': [0.1, 0.01],
        'min_child_weight': [1, 5]
    }

    param_grid = {
        'learning_rate':[0.3, 0.1, 0.6, 1],
        'n_estimators': [100, 200],
        'max_depth' : [6, 2, 12, 24],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1, 10],
        'reg_alpha' : [0, 0.1, 1, 2] 
    }

    # Create an XGBoost classifier
    xgb = XGBClassifier(random_state=123)

    df = pd.read_csv(settings.data_features2_csv)
    xgb = XGBClassifier(random_state=123)
    y_train, X_train = df.pop('winner'), df

    # Perform grid search with cross-validation
    grid_search = RandomizedSearchCV(xgb, param_grid, n_iter=30, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(best_params, '\n', best_score)



#model_tuning()
#compare_features_hypertune()
xgboost_tuning()