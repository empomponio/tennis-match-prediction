from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from sklearn.dummy import DummyClassifier


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
        # bet_on_better_odds(probs_pred, y_test)


def evaluate_baseline():
    df = pd.read_csv(settings.data_features_test_csv)
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    y = df.pop('winner')
    X = df
    clf = clf.fit(X, y)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, target_names=['Higher rank won', 'Lower rank won']))

def get_challenger_predictions():
    random.seed(123)
    df = pd.read_csv(settings.data_original_csv)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')  
    df = df[df.tourney_date.dt.year == settings.end_year] # test set
    df['prediction'] = 0
    # winner has lower odds -> winner is predicted as winner -> winner is ranked lower 
    df.loc[(df.winner_odds < df.loser_odds) & (df.winner_rank > df.loser_rank), 'prediction'] = 1
    # loser has lower odds -> loser is predicted as winner -> loser is ranked lower
    df.loc[(df.loser_odds < df.winner_odds) & (df.loser_rank > df.winner_rank), 'prediction'] = 1
    # winner and loser have same odds: prediction is random
    df.loc[df.loser_odds == df.winner_odds, 'prediction'] = random.randint(0, 1)
    return df.prediction.array

get_challenger_predictions()


def evaluate_challenger():
    df = pd.read_csv(settings.data_features_test_csv)
    y = df.pop('winner')
    y_pred = get_challenger_predictions()
    print(classification_report(y, y_pred, target_names=['Higher rank won', 'Lower rank won'], digits=4))
    print(type(y_pred))
    bet_on_all(y_pred, y)


"""
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
        df = pd.read_csv(settings.data_features_csv)
        df_test = pd.read_csv(settings.data_features_test_csv)
        evaluate_model(df, df_test, model, name)
"""


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



# bet on every single match according to the winner given by the model
def bet_on_all(y_pred, y_test, odds):
  print('Betting on all possible bets:')
  returns = 0

  if(len(odds) != len(y_pred) or len(y_pred) != len(y_test)):
    print(f'ERROR: arrays are of different lengths. odds: {len(odds)}, y_pred: {len(y_pred)}, y_test: {len(y_test)}')
    return

  for i in range(len(odds)):
    if y_pred[i] == y_test[i]: 
      returns += odds[i] - 1
    else:
      returns -= 1

  bets = len(y_pred)
  ret_bet = returns/bets
  ret_tot_bet = returns/len(y_pred)
  print([returns, bets, ret_bet, ret_tot_bet])


# bet on a match every time the class predicted by y_test matches pred_class
# pred_class can be 1 (lower ranked player wins) or 0 (higher ranked player wins)
def bet_on_class(y_pred, y_test, odds, pred_class=1):
  print(f'Betting on all possible bets for predicted class = {pred_class}:')
  returns = 0
  bets = 0

  if(len(odds) != len(y_pred) or len(y_pred) != len(y_test)):
    print(f'ERROR: arrays are of different lengths. odds: {len(odds)}, y_pred: {len(y_pred)}, y_test: {len(y_test)}')
    return

  for i in range(len(odds)):
    if y_pred[i] == pred_class:  # bet on upset
      bets += 1
      if y_pred[i] == y_test[i]: 
        returns += odds[i] - 1
      else:
        returns -= 1

  ret_bet = returns/bets
  ret_tot_bet = returns/len(y_pred)
  print([returns, bets, ret_bet, ret_tot_bet])


def bet_on_expectation(probs_pred, y_test, winner_odds, loser_odds):
    print(f'Betting on expected gain:')

    if(len(winner_odds) != len(loser_odds) or len(loser_odds) != len(probs_pred) or len(probs_pred) != len(y_test)):
        print(f'ERROR: arrays are of different lengths. winner_odds: {len(winner_odds)}, loser_odds: {len(loser_odds)}, probs_pred: {len(probs_pred)}, y_test: {len(y_test)}')
        return

    returns = 0
    bets = 0

    for i in range(len(winner_odds)):
        # odds for higher ranked player winning the match
        odds_0 = winner_odds[i] if y_test[i] == 0 else loser_odds[i]
        # odds for lower ranked player winning the match
        odds_1 = loser_odds[i] if y_test[i] == 0 else winner_odds[i]

        # expected value for bet on class 0
        exp_0 = probs_pred[i][0] * odds_0
        # expected value for bet on class 1
        exp_1 = probs_pred[i][1] * odds_1

        #print(f'odds_0: {odds_0}, odds_1: {odds_1}, probs_0: {probs_pred[i][0]}, probs_1: {probs_pred[i][1]}, exp_0: {exp_0}, exp_1: {exp_1}')

        if exp_0 > 1 and exp_0 > exp_1:
            #print('Bet on 0')
            bets +=1
            if y_test[i] == 0: 
                returns += winner_odds[i] - 1
            else:
                returns -= 1
        elif exp_1 > 1 and exp_1 > exp_0:
            #print('Bet on 1')
            bets +=1 
            if y_test[i] == 1: 
                returns += winner_odds[i] - 1
            else:
                returns -= 1 

    ret_bet = returns/bets
    ret_tot_bet = returns/len(probs_pred)
    print([returns, bets, ret_bet, ret_tot_bet])


def plot_bets(y_pred, y_eval, winner_odds):
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot()
    ax.set_yscale('log')
    ax.set_xlabel('Bet index')
    ax.set_ylabel('Betting odds')

    for i in range(len(y_pred)):
        marker = 'o' if y_eval[i] == 0 else '*'
        c = 'green' if y_pred[i] == y_eval[i] else 'red'
        y = winner_odds[i]
        x = i
        plt.scatter(x=x, y=y, marker=marker, c=c)

    plt.show()


# predict on evaluation set
def evaluate_metrics(models_list, X_train, y_train, X_eval, y_eval, winnerodds_eval, loserodds_eval):
    for _, (clf, model_name, fit_params) in enumerate(models_list):
        print(model_name)
        clf.fit(X_train, y_train, **fit_params)
        y_pred = clf.predict(X_eval)
        print(classification_report(y_eval, y_pred, digits=4))
        bet_on_all(y_pred, y_eval, winnerodds_eval)
        bet_on_class(y_pred, y_eval, winnerodds_eval, pred_class=1)
        bet_on_class(y_pred, y_eval, winnerodds_eval, pred_class=0)
        probs_pred = clf.predict_proba(X_eval) 
        bet_on_expectation(probs_pred, y_eval, winnerodds_eval, loserodds_eval)
        plot_bets(y_pred, y_eval, winnerodds_eval)

def evaluate_randomforest(X_train, y_train, X_eval, y_eval, winnerodds_train, winnerodds_eval, loserodds_eval):
    clf_prec1 = RandomForestClassifier(n_estimators=100, min_samples_split=10, max_features='log2', max_depth=3, criterion='entropy', bootstrap=True, random_state=123)
    clf_prec1_w = RandomForestClassifier(n_estimators=1000, min_samples_split=5, max_features=None, max_depth=None, criterion='entropy', bootstrap=True, random_state=123)

    clf_prec0 =RandomForestClassifier(n_estimators=100, min_samples_split=10, max_features=None, max_depth=None, criterion='gini', bootstrap=True, random_state=123)
    clf_prec0_w = RandomForestClassifier(n_estimators=1000, min_samples_split=10, max_features='sqrt', max_depth=3, criterion='entropy', bootstrap=True, random_state=123)

    clf_acc = RandomForestClassifier(n_estimators=1000, min_samples_split=10, max_features='log2', max_depth=10, criterion='entropy', bootstrap=True, random_state=123)
    clf_acc_w = RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features='sqrt', max_depth=None, criterion='entropy', bootstrap=True, random_state=123)

    params_list = {'sample_weight' : winnerodds_train}

    models_list = [
        (clf_prec1, 'Hypertuning for precision on class 1', {}),
        (clf_prec1_w, 'Hypertuning for precision on class 1 (weighted)', params_list),
        (clf_prec0, 'Hypertuning for precision on class 0', {}),
        (clf_prec0_w, 'Hypertuning for precision on class 0 (weighted)', params_list),
        (clf_acc, 'Hypertuning for accuracy', {}),
        (clf_acc_w, 'Hypertuning for accuracy (weighted)', params_list)
    ]
    evaluate_metrics(models_list, X_train, y_train, X_eval, y_eval, winnerodds_eval, loserodds_eval)

def model_evaluation():
    df = pd.read_csv(settings.data_features_csv)
    y = df.pop('winner').array
    X = df

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=.1, stratify=y, random_state=123)
    winnerodds_train = X_train.pop('winner_odds').array
    X_train.pop('loser_odds')
    winnerodds_eval = X_eval.pop('winner_odds').array
    loserodds_eval = X_eval.pop('loser_odds').array
    evaluate_randomforest(X_train, y_train, X_eval, y_eval, winnerodds_train, winnerodds_eval, loserodds_eval)

model_evaluation()
model_evaluation_advanced()



