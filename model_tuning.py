import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import settings, model_evaluation



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

#model_tuning()
#best_gbt()
best_adaboost()