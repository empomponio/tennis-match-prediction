import random
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

import settings


def winner_higher_rank(X, y):
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf = clf.fit(X, y)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, digits=4))


def winner_lower_odds(X, y):
    random.seed(settings.rnd_seed)
    y_pred = []
    for i in range(len(X)):
        if X.loc[i, 'p0_odds'] <  X.loc[i, 'p1_odds']:
            y_pred.append(0)
        elif X.loc[i, 'p1_odds'] <  X.loc[i, 'p0_odds']:
            y_pred.append(1)
        else: # odds are equal: prediction is random
            y_pred.append(random.randint(0, 1))
    print(classification_report(y, y_pred, digits=4))


def evaluate_baselines():
    df = pd.read_csv(settings.data_baseline_test_csv)
    y = df.pop('winner')
    X = df
    print('*********************************************************************')
    print('Scores for baseline (winner is the higher-ranked player)')
    print('*********************************************************************')
    winner_higher_rank(X, y)
    print('*********************************************************************')
    print('Scores for challenger (winner is the player with lower odds)')
    print('*********************************************************************')
    winner_lower_odds(X, y)


