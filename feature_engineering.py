import pandas as pd
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier


import settings

from model_tuning import get_keras_nn
from keras.optimizers import Adam

def find_best_features(X, y, clf, model_name):
    print(f'\nFeature elimination for {model_name}')
    min_features_to_select = 10    # Minimum number of features to consider
    cv = StratifiedKFold(5)
    X = StandardScaler().fit_transform(X)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring='accuracy',
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    print(f'Optimal number of features: {rfecv.n_features_}')
    n_scores = len(rfecv.cv_results_['mean_test_score'])
    plt.figure()
    plt.xlabel('Number of features selected')
    plt.ylabel('Mean test accuracy')
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_['mean_test_score'],
        yerr=rfecv.cv_results_['std_test_score'],
    )
    filename = f'{settings.path_plots}/{model_name}_fnum'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved fig into {filename}')
    return rfecv.ranking_


def feature_elimination(X, y, clf, model_name):
    ranking = find_best_features(X, y, clf, model_name)
    features = X.columns
    lowest_rank = np.max(ranking)
    feature_rank_dict = {}
    for i in range(len(ranking)):
        rank = ranking[i]
        if rank not in feature_rank_dict:
            feature_rank_dict[rank] = [features[i]]
        else:
            feature_rank_dict[rank].append(features[i])
    feature_rank_dict = sorted(feature_rank_dict.items())

    labels = []
    bars = []
    for rank, arr in feature_rank_dict:
      for label in arr:
        labels.append(label)
        bars.append(lowest_rank+1-rank)

    figsize = (lowest_rank, int(len(labels)*0.2))
    plt.figure(figsize=figsize)
    plt.barh(labels, bars)
    plt.gca().invert_yaxis()
    xticks = range(1, lowest_rank+1)
    plt.xticks(xticks, xticks[::-1])
    plt.xlabel('Feature Rank')
    filename = f'{settings.path_plots}/{model_name}_frank'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved fig into {filename}')


def feature_importance(X, y, forest, model_name):
    print('\nFeature importance with a Random Forest Classifier')
    feature_names = X.columns
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    best_features_series = forest_importances.sort_values(ascending=False)
    print(best_features_series)
    filename = f'{settings.path_plots}/{model_name}_fimp'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved fig into {filename}')


class OptimizedModel:
    def __init__(self, df, rf, xgb, name):
        self.y = df.pop('winner')
        self.X = df
        self.name = name 
        self.rf = rf
        self.xgb = xgb



def foo():
    # symmetric features, players ordered by rank
    df1 = pd.read_csv(settings.data_symmetric_csv)
    rf1 = RandomForestClassifier(random_state=123, criterion='entropy', max_depth=10, max_features='log2', min_samples_split=2, n_estimators=500)
    xgb1 = XGBClassifier(random_state=123, reg_lambda=1, reg_alpha=0.1, n_estimators=200, max_depth=2, learning_rate=0.1, gamma=0, booster='gbtree')
    #nn1 = KerasClassifier(random_state=123, optimizer__learning_rate=0.001)
    model1 = OptimizedModel(df1, rf1, xgb1, 'symmetric')
    nn1 = get_keras_nn(dropout_rate=0.3, neurons=512, n_features=len(model1.X.columns), learning_rate=0.01)
    KerasClassifier(model=nn1, epochs=50, batch_size=512)
    #{'optimizer__learning_rate': 0.01, 'model__neurons': 512, 'model__dropout_rate': 0.30000000000000004, 'epochs': 50, 'batch_size': 512}
    # double features, players ordered by rank
    df2 = pd.read_csv(settings.data_double_csv)
    rf2 = RandomForestClassifier(random_state=123, criterion='gini', max_depth=None, max_features='log2', min_samples_split=10, n_estimators=500)
    xgb2 = XGBClassifier(random_state=123, reg_lambda=0, reg_alpha=0, n_estimators=500, max_depth=2, learning_rate=0.1, gamma=1, booster='dart')
    model2 = OptimizedModel(df2, rf2, xgb2, 'double')
    #{'optimizer__learning_rate': 1.0, 'model__neurons': 256, 'model__dropout_rate': 0.1, 'epochs': 50, 'batch_size': 32}
    # symmetric features, players ordered randomly
    df3 = pd.read_csv(settings.data_random_csv)
    rf3 = RandomForestClassifier(random_state=123, criterion='gini', max_depth=10, max_features=None, min_samples_split=5, n_estimators=500)
    xgb3 = XGBClassifier(random_state=123, reg_lambda=1, reg_alpha=1, n_estimators=200, max_depth=6, learning_rate=0.1, gamma=10, booster='gbtree')  
    model3 = OptimizedModel(df3, rf3, xgb3, 'random')
    #{'optimizer__learning_rate': 0.001, 'model__neurons': 128, 'model__dropout_rate': 0.5, 'epochs': 50, 'batch_size': 32}

    feature_elimination(model1.X, model1.y, nn1, f'nn_{model1.name}')

    for model in [model1, model2, model3]:
        feature_elimination(model.X, model.y, model.rf, f'rf_{model.name}')
        feature_importance(model.X, model.y, model.rf, f'rf_{model.name}')
        feature_elimination(model.X, model.y, model.xgb, f'xgb_{model.name}')

foo()