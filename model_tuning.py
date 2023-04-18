from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, make_scorer, precision_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
import time
import math

import settings



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
    scoring_list = ['precision', make_scorer(precision_score, pos_label=0), 'accuracy']
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


#model_tuning()
#best_gbt()
#best_adaboost()
#randomforest_tuning()
#adaboost_tuning()
#xgboost_tuning()

def model_tuning():
    # split data in training and test set
    # in this case, the training set is used for hyperparameter tuning with cross validation
    # the test set is used with the best scoring models for more accurate metric evaluation and for cost evaluation
    df = pd.read_csv(settings.data_features_csv)
    y = df.pop('winner')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, stratify=y, random_state=123)
    odds_train = X_train.pop('winner_odds')
    X_train.pop('loser_odds')
    odds_test = X_test.pop('winner_odds')
    #logisticregression_tuning(X_train, y_train, odds_train)
    randomforest_tuning(X_train, y_train, odds_train)

model_tuning()