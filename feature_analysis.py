import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np


import settings


def find_best_features(X, y, clf, model_name):
    print('\nFeature elimination for', model_name)
    min_features_to_select = 1    # Minimum number of features to consider
    cv = StratifiedKFold(5)
    X = StandardScaler().fit_transform(X)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
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
    plt.title(f'Recursive Feature Elimination \nwith correlated features for {model_name}')
    plt.show()
    return rfecv.ranking_

def feature_elimination(X, y):
    lr = LogisticRegression(random_state=123)
    svc = LinearSVC(random_state=123, dual=False)
    models = {'Logistic Regression':lr, 'LinearSVC':svc}

    for name, model in models.items():
        ranking = find_best_features(X, y, model, name)
        features = X.columns
        feature_rank_dict = {}
        for i in range(len(ranking)):
            rank = ranking[i]
            if rank not in feature_rank_dict:
                feature_rank_dict[rank] = [features[i]]
            else:
                feature_rank_dict[rank].append(features[i])
        print(sorted(feature_rank_dict.items()))

def feature_importances(X, y):
    print('\nComputing feature importance with a Random Forest Classifier')
    feature_names = X.columns
    forest = RandomForestClassifier(random_state=123)
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
    print('Features ordered by importance:\n', best_features_series)


def feature_correlation(X):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()


def feature_analysis():
    df = pd.read_csv(settings.data_features_csv)
    y = df.pop('winner')
    X = df
    feature_elimination(X, y)
    feature_importances(X, y)
    feature_correlation(X)

def create_files_best_features():
    best_features = [  
        'rank_diff',
        'rank_points_logdiff',
        'height_diff',
        'age_diff',	
        'played_diff_year',	
        'winperc_diff_year',
        'played_diff_month',	
        'winperc_diff_month',	
        'played_diff_surface',	
        'winperc_diff_surface',	
        'played_diff_level',	
        'winperc_diff_level',	
        'winpercdiff_tourney',	
        'winperc_matchup',		
        'serve2perc_diff',	
        'return2perc_diff',	
        'bpsavedperc_pdiff',	
        'matchduration_diff',	
        'servereturn_diff'   
        ]
    cols = best_features + ['winner']

    df = pd.read_csv(settings.data_features_csv, usecols=cols)[cols]
    df_test = pd.read_csv(settings.data_features_test_csv, usecols=cols)[cols]
    df.to_csv(settings.data_features_hp_csv, index=False)
    df_test.to_csv(settings.data_features_hp_test_csv, index=False)


#feature_analysis()
create_files_best_features()