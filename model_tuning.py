from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score

import pandas as pd
import numpy as np
import time
import math

import settings
from neural_networks import nn_preprocessing, get_nn_grid



def hypertune_model(X, y, scoring_list, clf_name, search):
    results = []
    variables = ['scoring', 'best_score', 'best_params', 'running_time']
    for scoring in scoring_list:
        print('*****************************************************************************************************')    
        print(f'Hypertuning {clf_name} for {scoring} on 5-fold cross validation')   
        print('*****************************************************************************************************')       
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


def ml_tuning():
    # hypertuning on accuracy only
    # add more scores in the list to hypertune on other metrics
    scoring_list = ['accuracy']

    df1 = pd.read_csv(settings.data_symmetric_csv)
    df2 = pd.read_csv(settings.data_double_csv)

    # split data in training and test set
    # in this case, the training set is used for hyperparameter tuning with cross validation
    # the test set is used with the best scoring models for more accurate metric evaluation and for cost evaluation
    n_jobs = -1
    cv = StratifiedKFold()
    n_iter = 100

    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=settings.rnd_seed))
    lr_grid = [
        {
            'logisticregression__max_iter' : [100, 1000],
            'logisticregression__penalty' : ['l2', 'none'],
            'logisticregression__C' : np.logspace(-10, 9, 20),
            'logisticregression__solver' : ['lbfgs','newton-cholesky','saga'],
        }
    ]
    lr_search = GridSearchCV(lr, lr_grid, n_jobs=n_jobs, cv=cv)

    rf =   RandomForestClassifier(random_state=settings.rnd_seed)
    rf_grid = {
        'n_estimators' : [10, 100, 500],
        'max_depth': [None, 3, 10],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    rf_search = GridSearchCV(rf, rf_grid,  n_jobs=n_jobs, cv=cv)

    xgb = XGBClassifier(random_state=settings.rnd_seed)

    xgb_grid =  {
        'learning_rate':[0.3, 0.1, 0.6],
        'n_estimators': [100, 200],
        'max_depth' : [6, 2, 12],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1],
        'reg_alpha' : [0, 0.1, 1]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_grid,  n_jobs=n_jobs, cv=cv, n_iter=n_iter)

    clf_list = [
        (lr_search, 'logistic_regression'),
        (rf_search, 'random_forest'),
        (xgb_search, 'xgboost'),
    ]

    y1 = df1.pop('winner')
    y2 = df2.pop('winner')

    df_list = [
        (df1, y1, 'symmetric'), 
        (df2, y2, 'double'), 
    ]

    for (clf_search, clf_name) in clf_list:
        for (X, y, df_name) in df_list:
            print('Tuning', clf_name)
            hypertune_model(X=X, y=y, 
                            scoring_list=scoring_list, 
                            search=clf_search, 
                            clf_name=f'{clf_name}_{df_name}')


def nn_tuning():
    df1 = pd.read_csv(settings.data_symmetric_csv)
    y1 = df1.pop('winner')
    cat_sym = ['wildcard', 'hand_matchup']
    df1 = nn_preprocessing(df1, cat_sym)

    df2 = pd.read_csv(settings.data_double_csv)
    y2 = df2.pop('winner')
    cat_double = ['p0_hand', 'p1_hand']
    df2 = nn_preprocessing(df2, cat_double)

    df_list = [
        (df1, y1, 'symmetric'),
        (df2, y2, 'double')
    ]

    n_jobs = -1
    cv = StratifiedKFold()
    n_iter = 1

    for (X, y, df_name) in df_list:
        clf_name = 'neural_network'
        print('Tuning', clf_name)
        (nn, nn_grid) = get_nn_grid(len(X.columns))
        search = RandomizedSearchCV(nn, nn_grid,  n_jobs=n_jobs, cv=cv, n_iter=n_iter, random_state=settings.rnd_seed)
        hypertune_model(X=X, y=y, 
                        scoring_list=['accuracy'],
                        search=search, 
                        clf_name=f'{clf_name}_{df_name}')


def xgb_tuning_fscore():
    scoring_list = [make_scorer(fbeta_score, beta=0.5)]
    df = pd.read_csv(settings.data_symmetric_csv)
    n_jobs = -1
    cv = StratifiedKFold()
    xgb = XGBClassifier(random_state=settings.rnd_seed)
    n_iter = 100

    # same parameter as the other search, with last row for weighting on class 1
    xgb_grid =  {
        'learning_rate':[0.3, 0.1, 0.6],
        'n_estimators': [100, 200],
        'max_depth' : [6, 2, 12],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1],
        'reg_alpha' : [0, 0.1, 1],
        'scale_pos_weight' : [10, 25, 50, 75, 99, 100, 1000]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_grid,  n_jobs=n_jobs, cv=cv, n_iter=n_iter)
    y = df.pop('winner')
    X = df
    hypertune_model(X=X, y=y, scoring_list=scoring_list, search=xgb_search, clf_name='xgboost_f1_symmetric')




def hypertune_all():
    #ml_tuning()
    #nn_tuning()
    xgb_tuning_fscore()

