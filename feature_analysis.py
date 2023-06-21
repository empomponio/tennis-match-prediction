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

import settings




def find_best_features(X, y, clf, model_name):
    print('*****************************************************************************************************')    
    print(f'Feature elimination for {model_name} on a 5-fold cross validation')
    print('*****************************************************************************************************')    

    min_features_to_select = 1
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
    print('Feature importance with a Random Forest Classifier')
    feature_names = X.columns
    forest.fit(X, y)
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    best_features_series = forest_importances.sort_values(ascending=False)
    print('Features ordered by importance:\n', best_features_series)
    filename = f'{settings.path_plots}/{model_name}_fimp'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved fig into {filename}')



def get_tree_models():
    xgb = XGBClassifier(random_state=settings.rnd_seed, 
                        reg_lambda=1, 
                        reg_alpha=0.1, 
                        n_estimators=200, 
                        max_depth=2, 
                        learning_rate=0.1, 
                        gamma=0, 
                        booster='gbtree') 
    rf = RandomForestClassifier(random_state=settings.rnd_seed, 
                                  criterion='entropy', 
                                  max_depth=10, 
                                  max_features='log2', 
                                  min_samples_split=2, 
                                  n_estimators=500)
    models = [(xgb, 'xgb_symmetric'), (rf, 'rf_symmetric')]
    return models

def get_rf():
    return 


def analyze_features():
    df = pd.read_csv(settings.data_symmetric_csv)
    y = df.pop('winner')
    X = df

    models = get_tree_models()
    for (clf, name) in models:
      feature_elimination(X, y, clf, name)
      feature_importance(X, y, clf, name)