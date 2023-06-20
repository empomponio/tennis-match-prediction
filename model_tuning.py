from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
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


def get_keras_nn(dropout_rate, neurons, n_features, learning_rate):
    model = models.Sequential()
    model.add(Dense(units=neurons, activation='relu', input_shape=(n_features,)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    print(keras.optimizers.Adam.build)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


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


def model_tuning():

    #scoring_list = ['precision', 'f1', 'accuracy']
    #scoring_list = ['f1']
    scoring_list = [make_scorer(fbeta_score, beta=0.5)]


    df1 = pd.read_csv(settings.data_symmetric_csv)
    df2 = pd.read_csv(settings.data_double_csv)
    df3 = pd.read_csv(settings.data_random_csv)

    # split data in training and test set
    # in this case, the training set is used for hyperparameter tuning with cross validation
    # the test set is used with the best scoring models for more accurate metric evaluation and for cost evaluation
    n_jobs = -1
    cv = StratifiedKFold()

    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=123))
    lr_grid = [
        {
            'logisticregression__max_iter' : [100, 1000],
            'logisticregression__penalty' : ['l2', 'none'],
            'logisticregression__C' : np.logspace(-10, 9, 20),
            'logisticregression__solver' : ['lbfgs','newton-cholesky','saga'],
        }
    ]
    lr_search = GridSearchCV(lr, lr_grid, n_jobs=n_jobs, cv=cv)

    rf =   RandomForestClassifier(random_state=123)
    rf_grid = {
        'n_estimators' : [10, 100, 500],
        'max_depth': [None, 3, 10],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    rf_search = GridSearchCV(rf, rf_grid,  n_jobs=n_jobs, cv=cv)

    xgb = XGBClassifier(random_state=123)

    xgb_grid =  {
        'learning_rate':[0.3, 0.1, 0.6],
        'n_estimators': [100, 200],
        'max_depth' : [6, 2, 12],
        'booster' : ['gbtree', 'gblinear', 'dart'],  
        'gamma' : [0, 1, 10],   
        'reg_lambda' : [1, 0, 0.1],
        'reg_alpha' : [0, 0.1, 1],
        #'scale_pos_weight' : [10, 25, 50, 75, 99, 100, 1000]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_grid,  n_jobs=n_jobs, cv=cv, n_iter=100)

    clf_list = [
        #(lr_search, 'logistic_regression'),
        #(rf_search, 'random_forest'),
        (xgb_search, 'xgboost_f1'),
    ]

    y1 = df1.pop('winner')
    y2 = df2.pop('winner')
    y3 = df3.pop('winner')

    df_list = [
        (df1, y1, 'symmetric'), 
        #(df2, y2, 'double'), 
        #(df3, y3, 'random')
    ]

    for (clf_search, clf_name) in clf_list:
        for (X, y, df_name) in df_list:
            print('Tuning', clf_name)
            hypertune_model(X=X, y=y, 
                            scoring_list=scoring_list, 
                            search=clf_search, 
                            clf_name=f'{clf_name}_{df_name}')
    #nn_tuning(df_list)


def nn_tuning(df_list):
    n_jobs = -1
    cv = StratifiedKFold()
    n_iter = 100

    for (X, y, df_name) in df_list:
        clf_name = 'neural_network'
        print('Tuning', clf_name)
        (nn, nn_grid) = get_neuralnetwork_classifier(len(X.columns))
        search = RandomizedSearchCV(nn, nn_grid,  n_jobs=n_jobs, cv=cv, n_iter=n_iter, random_state=123)
        hypertune_model(X=X, y=y, 
                        scoring_list=['accuracy'],
                        search=search, 
                        clf_name=f'{clf_name}_{df_name}')



model_tuning()