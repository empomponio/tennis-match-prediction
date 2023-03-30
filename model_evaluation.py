import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier

import settings

# bet on every single match according to the winner given by the model
def bet_on_all(y_pred, y_test):
    # test on the csv for the last year
    df = pd.read_csv(settings.data_original_csv)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')    
    df = df[df.tourney_date.dt.year == settings.end_year] # test set
    odds = df['winner_odds']

    # baseline 2: winner is the highest ranked player
    correct = 0
    returns = 0

    if(len(odds) != len(y_pred) or len(y_pred) != len(y_test)):
        print(f'ERROR: arrays are of different lengths. odds: {len(odds)}, y_pred: {len(y_pred)}, y_test: {len(y_test)}')
        return

    for i in range(len(odds)):
        if y_pred[i] == y_test[i]: 
            correct += 1
            returns += odds.iloc[i] - 1
        else:
            returns -= 1

    print(f'Returns: {returns} on {len(df)} bets\nAverage return per bet: {returns/len(df)}')

# bet on the winner or the loser if the respective odds given by the model are better than the implied odds of the bookmaker
def bet_on_better_odds(probs_pred, y_test):
    bets, returns = 0, 0
    df = pd.read_csv(settings.data_original_csv)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')    
    df = df[df.tourney_date.dt.year == settings.end_year] # test set
    if(len(df) != len(probs_pred) or len(probs_pred) != len(y_test)):
        print(f'ERROR: arrays are of different lengths. odds: {len(df)}, y_pred: {len(probs_pred)}, y_test: {len(y_test)}')
        return    
    
    for i, probs in enumerate(probs_pred):
        # initialize probs as if player1 was the winner
        # if player 2 was the winner, switch probs instead
        winner_prob, loser_prob = probs[0], probs[1]     
        if y_test[i] == 2:
            winner_prob, loser_prob = loser_prob, winner_prob    
        # get implied odds
        winner_implied_odds, loser_implied_odds = 1/df.iloc[i].winner_odds, 1/df.iloc[i].loser_odds
        # check if the probabilities of the model is greater than the one of the implied odds
        #print(f'winner_prob: {winner_prob}, loser_prob:{loser_prob}, winner_implied_odds: {winner_implied_odds}, loser_implied_odds: {loser_implied_odds}')
        if winner_prob > winner_implied_odds:
            # bet on the winner
            bets += 1
            returns += df.iloc[i].winner_odds - 1
            #print(f'Bets: {bets}, Returns: {returns}\n')
        elif loser_prob > loser_implied_odds:
            # bet on the loser
            bets += 1
            returns -= 1
            #print(f'Bets: {bets}, Returns: {returns}\n')

    print(f'Returns: {returns} on {bets} bets\nAverage return per bet: {returns/bets}\n\n')        
    return

def evaluate_model(df, df_test, clf, model_name):
    # Accuracy with cross validation
    y = df.pop('winner')
    X = df
    print(f'\n\n{model_name} for features:\n {df.columns}:')
    cv_scores = cross_validate(clf, X, y, cv=10)
    print('Cross validation scores:', cv_scores['test_score'], '\nAverage: ', np.average(cv_scores['test_score']))

    # Accuracy on test set
    y_test = df_test.pop('winner')
    X_test = df_test
    clf.fit(X, y)
    test_score = clf.score(X_test, y_test)
    print(f'Accuracy on test set: {test_score}')

    # Returns on test set with two simple betting strategies
    y_pred = clf.predict(X_test)
    bet_on_all(y_pred, y_test)
    if model_name != 'Support Vector Machine':
        probs_pred = clf.predict_proba(X_test)
        bet_on_better_odds(probs_pred, y_test)


def model_evaluation():
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=123))
    svc = make_pipeline(StandardScaler(), LinearSVC(random_state=123, dual=False))
    dtc = DecisionTreeClassifier(random_state=123)
    rfc = RandomForestClassifier(random_state=123)

    models = {
        'Logistic Regression': lr, 
        'Support Vector Machine': svc,
        'Decision Tree Classifier': dtc,
        'Random Forest Classifier': rfc
    }

    for name, model in models.items():
        # label encoding
        df_le = pd.read_csv(settings.data_features_le_csv)
        df_le_test = pd.read_csv(settings.data_features_le_test_csv)

        # one-hot encoding
        df_ohe = pd.read_csv(settings.data_features_onehot_csv)
        df_ohe_test = pd.read_csv(settings.data_features_onehot_test_csv)

        evaluate_model(df_le, df_le_test, model, name)
        evaluate_model(df_ohe, df_ohe_test, model, name)


def model_evaluation_advanced():
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=123))
    svc = make_pipeline(StandardScaler(), LinearSVC(random_state=123, dual=False))
    dtc = DecisionTreeClassifier(random_state=123)
    rfc = RandomForestClassifier(random_state=123)
    etc = ExtraTreesClassifier(random_state=123)
    abc = AdaBoostClassifier(random_state=123)
    gbt = HistGradientBoostingClassifier(random_state=123)

    models = {
        'Logistic Regression': lr, 
        'Support Vector Machine': svc,
        'Decision Tree Classifier': dtc,
        'Random Forest Classifier': rfc,
        'Extra Trees Classifier': etc,
        'AdaBoost Classifier': abc,
        'Gradient Boosting Classifier': gbt
    }

    for name, model in models.items():
        # all features
        df_all = pd.read_csv(settings.data_features_csv)
        df_all_test = pd.read_csv(settings.data_features_test_csv)

        # hand-picked features
        df = pd.read_csv(settings.data_features_hp_csv)
        df_test = pd.read_csv(settings.data_features_hp_test_csv)

        evaluate_model(df_all, df_all_test, model, name)
        evaluate_model(df, df_test, model, name)


#model_evaluation()
model_evaluation_advanced()




