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


def logistic_regression_tuning():
    print('\n\n\nHypertuning LogisticRegression')
    parameters = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'newton-cholesky'],
        'tol' : [10**-7, 10**-4, 10**-2],
        'max_iter' : [100000]
        }
    clf = GridSearchCV(LogisticRegression(random_state=123), parameters, cv=10, n_jobs=-1, verbose=1)
    df = pd.read_csv(settings.data_features_hp_csv)
    y = df.pop('winner')
    X = df
    clf.fit(X, y)
    print(clf.score(X, y))
    print(clf.best_params_)




def gradient_boosting_tuning():
    print('Hypertuning HistGradientBoostingClassifier')
    parameters = {
        'loss':["log_loss"],
        'learning_rate': [0.05, 0.1, 0.5, 1],
        'max_iter': [100, 1000, 10000],
        'max_leaf_nodes' : [5, 10, 31, None],
        'min_samples_leaf' : [20, 40, 80],
        'tol' : [10**-7, 10**-4, 10**-2]
        }
    clf = GridSearchCV(HistGradientBoostingClassifier(random_state=123), parameters, cv=10, n_jobs=-1, verbose=1)
    df = pd.read_csv(settings.data_features_hp_csv)
    y = df.pop('winner')
    X = df
    clf.fit(X, y)
    print(clf.score(X, y))
    print(clf.best_params_)

def model_tuning():
    logistic_regression_tuning()
    adaboost_tuning()
    gradient_boosting_tuning()


def best_gbt():
    clf = HistGradientBoostingClassifier(
        random_state=123, 
        learning_rate=0.1, 
        loss='log_loss',
        max_iter=100, 
        max_leaf_nodes=10, 
        min_samples_leaf=80, 
        tol=1e-07
        )
    df = pd.read_csv(settings.data_features_hp_csv)
    df_test = pd.read_csv(settings.data_features_hp_test_csv)
    model_evaluation.evaluate_model(df, df_test, clf, 'HistGradientBoostingClassifier')



def best_adaboost():
    clf = AdaBoostClassifier(random_state=123, learning_rate=0.1, n_estimators=500)
    df = pd.read_csv(settings.data_features_hp_csv)
    df_test = pd.read_csv(settings.data_features_hp_test_csv)
    model_evaluation.evaluate_model(df, df_test, clf, 'HistGradientBoostingClassifier')

"""
def randomforest_tuning():
    #scorings = ['precision', make_scorer(precision_score, pos_label=0), 'accuracy']
    scorings = [make_scorer(precision_score, pos_label=0)]
    
    df = pd.read_csv(settings.data_features_csv)
    odds = df.pop('winner_odds')
    y = df.pop('winner')
    X = df

    param_grid = {
        "n_estimators" : [10, 100, 1000],
        "max_depth": [None, 3, 10],
        "max_features": ['sqrt', 'log2', None],
        "min_samples_split": [2, 5,10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }


    clf = RandomForestClassifier(random_state=123)

    #fit_params_list = [{}, {'sample_weight' : odds}]

    fit_params_list = [{'sample_weight' : odds}]


    for scoring in scorings:
        for fit_params in fit_params_list:
            search = RandomizedSearchCV(clf, param_grid, n_jobs=-1, scoring=scoring, n_iter=30)
            print(f'Starting search for {scoring}, {fit_params}')
            search.fit(X, y, **fit_params)
            print(f'Best parameters (score={search.best_score_})')
            print(search.best_params_)
"""

"""
def adaboost_tuning():
    print('\n\n\nHypertuning AdaBoostClassifier')
    parameters = {
        'learning_rate': [0.05, 0.1, 0.5, 1],
        'n_estimators': [10, 50, 100, 500],
        }
    clf = GridSearchCV(AdaBoostClassifier(random_state=123), parameters, cv=10, n_jobs=-1, verbose=1)
    df = pd.read_csv(settings.data_features_hp_csv)
    y = df.pop('winner')
    X = df
    clf.fit(X, y)
    print(clf.score(X, y))
    print(clf.best_params_)
"""

def adaboost_tuning():
    print('\n\n\nHypertuning AdaBoostClassifier')
    param_grid = {
        'learning_rate': np.logspace(-4, 1, 4),
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'base_estimator' : [None, DecisionTreeClassifier(max_depth=2), LogisticRegression()]
        }

    precision_0 = make_scorer(precision_score, pos_label=0)
    
    df = pd.read_csv(settings.data_features_csv)
    odds = df.pop('winner_odds')
    y = df.pop('winner')
    X = df

    clf = AdaBoostClassifier(random_state=123)

    #fit_params_list = [{}, {'sample_weight' : odds}]

    fit_params = [{'sample_weight' : odds}]

    for (scoring, fit_params) in [('precision', {}), (precision_0, fit_params), ('accuracy', {})]:
            search = RandomizedSearchCV(clf, param_grid, n_jobs=-1, scoring=scoring, n_iter=1)
            print(f'Starting search for scoring: {scoring}, fit_params: {bool(fit_params)}')
            search.fit(X, y, **fit_params)
            print(f'Best parameters (score={search.best_score_})')
            print(search.best_params_)

def xgboost_tuning():
    print('Hypertuning XGBClassifier')
    """
    param_grid = {
        'learning_rate': np.logspace(-4, 0, 4),
        'n_estimators': [20, 100, 200, 500],
        'max_depth' : [1, 2, 6, 10, 15],
        'booster' : ['gbtree', 'gblinear', 'dart']
        }
    """

    """    
    param_grid = {
        'learning_rate':[0.3, 0.1, 0.6, 1],
        'n_estimators': [20, 100, 200, 500],
        'max_depth' : [6, 2, 12, 24],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1, 10],
        'reg_alpha' : [0, 0.1, 1, 2] 
        }
    """

    param_grid = {}
    
    precision_0 = make_scorer(precision_score, pos_label=0)

    df = pd.read_csv(settings.data_features_csv)
    odds = df.pop('winner_odds')
    y = df.pop('winner')
    X = df

    #clf = XGBClassifier(random_state=123, objective='binary:logistic', eval_metric='logloss')
    clf = XGBClassifier(random_state=123)

    results = []
    variables = ['scoring', 'fit_params', 'best_score', 'best_params', 'running_time']
    scoring_list = ['precision', precision_0, 'accuracy']
    fit_params = {'sample_weight' : odds}
    fit_params_list = [{}, fit_params]

    for scoring in scoring_list:
        for fit_params in fit_params_list:
            start = time.time()
            search = RandomizedSearchCV(clf, param_grid, n_jobs=-2, scoring=scoring, n_iter=1)
            print(f'Starting search for scoring: {scoring}, fit_params: {bool(fit_params)}')
            search.fit(X, y, **fit_params)
            end = time.time()
            running_time = math.trunc(time.time() - start)
            results.append([scoring, bool(fit_params), search.best_score_,  search.best_params_, running_time])
            print(f'Best parameters (score={search.best_score_})')
            print(search.best_params_)
            print(f'Running time: {math.trunc(end-start)} seconds')
    results_df = pd.DataFrame(data=results, columns=variables)
    csv_name = f'{settings.path_hypertuning}/xgboost.csv'
    results_df.to_csv(csv_name, index=False)
    print('Saved results to', csv_name)


def randomforest_tuning(X, y, w):

    param_grid = {
        "n_estimators" : [10, 100, 1000],
        "max_depth": [None, 3, 10],
        "max_features": ['sqrt', 'log2', None],
        "min_samples_split": [2, 5,10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }

    clf = RandomForestClassifier(random_state=123)
    #scoring_list = ['precision', make_scorer(precision_score, pos_label=0), 'accuracy']
    scoring_list = ['f1', make_scorer(fbeta_score, beta=.5)]
    fit_params_list = [{}, {'sample_weight' : w}]

    results = []
    variables = ['scoring', 'fit_params', 'best_score', 'best_params', 'running_time']
    for scoring in scoring_list:
        for fit_params in fit_params_list:
            start = time.time()
            search = RandomizedSearchCV(clf, param_grid, n_jobs=-2, scoring=scoring, n_iter=30)
            print(f'Starting search for scoring: {scoring}, fit_params: {bool(fit_params)}')
            search.fit(X, y, **fit_params)
            end = time.time()
            running_time = math.trunc(time.time() - start)
            results.append([scoring, bool(fit_params), search.best_score_,  search.best_params_, running_time])
            print(f'Best parameters (score={search.best_score_})')
            print(search.best_params_)
            print(f'Running time: {math.trunc(end-start)} seconds')
    results_df = pd.DataFrame(data=results, columns=variables)
    csv_name = f'{settings.path_hypertuning}/randomforest.csv'
    results_df.to_csv(csv_name, index=False)
    print('Saved results to', csv_name)


def neuralnetwork_tuning(X, y, w):

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
        model.add(Dense(units=neurons, activation='relu', input_shape=(len(X.columns),)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    model = KerasClassifier(model=create_model, verbose=0)

    scoring_list = ['precision', make_scorer(precision_score, pos_label=0), 'accuracy', 'f1']
    sample_weights = [None, w]
    
    results = []
    variables = ['scoring', 'fit_params', 'best_score', 'best_params', 'running_time']

    for scoring in scoring_list:
        for sample_weight  in sample_weights:
            start = time.time()
            search = RandomizedSearchCV(model, param_grid, n_jobs=-1, n_iter=1, scoring=scoring)
            print(f'Starting search for scoring: {scoring}, sample_weight: {type(sample_weight)}')
            search.fit(X, y, sample_weight=sample_weight)
            end = time.time()
            running_time = math.trunc(time.time() - start)
            print(f'Best parameters (score={search.best_score_})')
            print(search.best_params_)
            print(f'Running time: {math.trunc(end-start)} seconds')
            results.append([scoring, type(sample_weight), search.best_score_, search.best_params_, running_time])

        results_df = pd.DataFrame(data=results, columns=variables)
        csv_name = f'{settings.path_hypertuning}/neuralnetwork.csv'
        results_df.to_csv(csv_name, index=False)
        print('Saved results to', csv_name)


def logisticregression_tuning(X, y, w):

    param_grid = [
        {
            'logisticregression__penalty' : ['l2', None],
            'logisticregression__C' : np.logspace(-10, 9, 20),
            'logisticregression__solver' : ['lbfgs','newton-cholesky','saga'],
        }
    ]

    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=123))
    
    scoring_list = ['precision', make_scorer(precision_score, pos_label=0), 'accuracy']
    fit_params_list = [{}, {'logisticregression__sample_weight' : w}]
    
    results = []
    variables = ['scoring', 'fit_params', 'best_score', 'best_params', 'running_time']
    for scoring in scoring_list:
        for fit_params in fit_params_list:
            start = time.time()
            search = GridSearchCV(clf, param_grid, n_jobs=-2, scoring=scoring)
            print(f'Starting search for scoring: {scoring}, fit_params: {bool(fit_params)}')
            search.fit(X, y, **fit_params)
            end = time.time()
            running_time = math.trunc(time.time() - start)
            results.append([scoring, bool(fit_params), search.best_score_, search.best_params_, running_time])
            print(f'Best parameters (score={search.best_score_})')
            print(search.best_params_)
            print(f'Running time: {math.trunc(end-start)} seconds')
    results_df = pd.DataFrame(data=results, columns=variables)
    csv_name = f'{settings.path_hypertuning}/logisticregression.csv'
    results_df.to_csv(csv_name, index=False)
    print('Saved results to', csv_name)


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

def model_tuning():

    #scoring_list = ['precision', 'f1', 'accuracy']
    scoring_list = [make_scorer(fbeta_score, beta=0.5)]

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
        'n_estimators': [20, 100, 200, 500],
        'max_depth' : [6, 2, 12, 24],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1, 10],
        'reg_alpha' : [0, 0.1, 1, 2] 
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_grid, n_jobs=-2, n_iter=20)

    (nn, nn_grid) = get_neuralnetwork_classifier(len(X.columns))
    nn_search = RandomizedSearchCV(nn, nn_grid, n_jobs=-2, n_iter=20)

    clf_list = [
        #(lr_search, 'logistic_regression'),
        (rf_search, 'random_forest'),
        (xgb_search, 'xgboost'),
        (nn_search, 'neural_network')
    ]

    for (clf_search, clf_name) in clf_list:
        hypertune_model_weighted(X=X, y=y, scoring_list=scoring_list, search=clf_search, clf_name=clf_name, sample_weight=w)


model_tuning()