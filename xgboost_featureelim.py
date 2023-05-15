from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

import settings


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


# 1. Find optimanl number of feature
# 2. t-test to compare accuracy with baseline
# 3. Analyze betting patterns and try to find a betting strategy

def find_best_features(X, y, clf, clf_name, scoring):
    print(f'\nFeature elimination for {clf_name} on {scoring}')
    min_features_to_select = 5    # Minimum number of features to consider
    cv = StratifiedKFold(10)
    X = StandardScaler().fit_transform(X)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring=scoring,
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
    plt.title(f'Recursive Feature Elimination \nwith correlated features for {clf_name}')
    plt.show()
    return rfecv.ranking_

def feature_elimination(X, y, clf, clf_name, scoring):
    ranking = find_best_features(X, y, clf, clf_name, scoring)
    features = X.columns
    feature_rank_dict = {}
    for i in range(len(ranking)):
        rank = ranking[i]
        if rank not in feature_rank_dict:
            feature_rank_dict[rank] = [features[i]]
        else:
            feature_rank_dict[rank].append(features[i])
    print(sorted(feature_rank_dict.items()))


def feature_analysis():
    (X, y, w) = get_train_x_y()
    xgb_acc = XGBClassifier(reg_lambda=1, reg_alpha=2, n_estimators=100, max_depth=2, learning_rate=0.1, gamma=10, booster='dart', random_state=123)
    lr_prec = LogisticRegression(C=0.0001, penalty='l2', solver='saga', random_state=123)


    clf_list = {
        #(lr_prec, 'Logistic Regression', 'precision'),
        (xgb_acc, 'XGBoost', 'accuracy')
    }
    for (clf, clf_name, scoring) in clf_list:
        feature_elimination(X=X, y=y, clf=clf, clf_name=clf_name, scoring=scoring)


feature_analysis()






