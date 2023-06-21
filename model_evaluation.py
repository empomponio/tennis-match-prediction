from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import settings
import os, random
import tensorflow as tf

from neural_networks import get_nn, nn_preprocessing


y_true_cv = []
y_pred_cv = []

def classification_report_cv(y_true, y_pred):
    y_true_cv.extend(y_true)
    y_pred_cv.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score

def reset_scores():
    global y_true_cv
    global y_pred_cv
    y_true_cv = []
    y_pred_cv = []


def get_xgb():
    columns = ['wins_diff_year', 'servereturn_diff', 'rank_points_diff', 'wins_diff_surface', 'played_diff_month', 'serveperc_diff', 'winperc_diff_surface', 'winpercdiff_matchup',
    'winperc_diff_year', 'played_diff_surface', 'wins_diff_month', 'momentum_diff', 'winpercdiff_tourney', 'age_diff', 'matchduration_diff', 'rank_diff']
    clf = XGBClassifier(random_state=settings.rnd_seed, 
                        reg_lambda=1, 
                        reg_alpha=0.1, 
                        n_estimators=200, 
                        max_depth=2, 
                        learning_rate=0.1, 
                        gamma=0, 
                        booster='gbtree') 
    return (clf, columns)

def get_rf():
    return RandomForestClassifier(random_state=settings.rnd_seed, 
                                  criterion='entropy', 
                                  max_depth=10, 
                                  max_features='log2', 
                                  min_samples_split=2, 
                                  n_estimators=500)
    

def evaluate_cv():
    df = pd.read_csv(settings.data_symmetric_csv)
    y = df.pop('winner')

    # XGBoost (reduced columns)
    print('*********************************************************************')
    print('Evaluating XGBoost (reduced columns) on a 10-fold cross validation')   
    print('*********************************************************************')
    reset_scores()
    (xgb, columns) = get_xgb()
    X_xgb = df[columns]
    cross_val_score(xgb, X=X_xgb, y=y, scoring=make_scorer(classification_report_cv), cv=10)   
    print(classification_report(y_true_cv, y_pred_cv, digits=4))

    # Neural Network
    print('*********************************************************************')
    print('Evaluating Neural Netowrk on a 10-fold cross validation')   
    print('*********************************************************************')    
    reset_scores()
    df_nn = pd.read_csv(settings.data_double_csv)
    df_nn.pop('winner')
    cat_double = ['p0_hand', 'p1_hand']
    X_nn = nn_preprocessing(df_nn, cat_double)
    nn = get_nn(len(X_nn.columns))
    cross_val_score(nn, X=X_nn, y=y, scoring=make_scorer(classification_report_cv), cv=10)   
    print(classification_report(y_true_cv, y_pred_cv, digits=4))

    # Random Forest
    print('*********************************************************************')
    print('Evaluating Random Forest on a 10-fold cross validation')   
    print('*********************************************************************')    
    reset_scores()
    rf = get_rf()
    X_rf = df
    cross_val_score(rf, X=X_rf, y=y, scoring=make_scorer(classification_report_cv), cv=10)   
    print(classification_report(y_true_cv, y_pred_cv, digits=4))


def evaluate_test():
    (clf, columns) = get_xgb()
    df = pd.read_csv(settings.data_symmetric_csv)
    y = df.pop('winner')
    X = df[columns]

    print('*********************************************************************')
    print('Fitting model to the training set...')   
    clf.fit(X=X, y=y)
    df_test = pd.read_csv(settings.data_symmetric_test_csv)
    y_test = df_test.pop('winner')
    X_test = df_test[columns]
    y_pred = clf.predict(X=X_test)
    print('*********************************************************************')
    print('Evaluation on the test set')
    print('*********************************************************************')
    print(classification_report(y_test, y_pred, digits=4))  


def evaluate_test_nn():

    df = pd.read_csv(settings.data_double_csv)
    y = df.pop('winner')
    cat_double = ['p0_hand', 'p1_hand']
    X = nn_preprocessing(df, cat_double)
    clf = get_nn(len(X.columns))

    print('*********************************************************************')
    print('Fitting model to the training set...')    
    clf.fit(X=X, y=y)
    df_test = pd.read_csv(settings.data_double_test_csv)
    y_test = df_test.pop('winner')
    X_test = nn_preprocessing(df_test, cat_double)
    y_pred = clf.predict(X=X_test)
    print('*********************************************************************')
    print('Evaluation on the test set')    
    print(classification_report(y_test, y_pred, digits=4))    

def evaluate_test_fscore():
    return


