import pandas as pd
import settings


def winner_lowest_odds(df):
    print('Baseline 1: winner is the player with the lowest odds')
    correct = 0
    returns = 0
    # calculating correct predictions and profits
    # winner with lower odds means prediction is correct
    # same odds for both player means a random bet would be correct half of the time
    # to calculate returns, assume an amount of 1 is bet on the lower odds every time
    for i in range(len(df)):
        if df.iloc[i].winner_odds < df.iloc[i].loser_odds:
            correct += 1
            returns += df.iloc[i].winner_odds - 1
        elif df.iloc[i].winner_odds == df.iloc[i].loser_odds:
            correct += 0.5
            returns += (df.iloc[i].winner_odds / 2) - 1
        else:
            returns -= 1
    accuracy = correct / len(df)
    print(f'Accuracy: {accuracy}\nReturns: {returns} on {len(df)} bets\nAverage return per bet: {returns/len(df)}\n')


def winner_highest_rank(df):
    print('Baseline 2: winner is the highest ranked player')
    correct = 0
    returns = 0
    # winner with lowest rank number (which means highest rank) means prediction is correct
    # only two options in this case because two players cannot have the same rank
    for i in range(len(df)):
        if df.iloc[i].winner_rank < df.iloc[i].loser_rank:
            correct += 1
            returns += df.iloc[i].winner_odds - 1
        else:
            returns -= 1
    accuracy = correct / len(df)
    print(f'Accuracy: {accuracy}\nReturns: {returns} on {len(df)} bets\nAverage return per bet: {returns/len(df)}')

def baselines():
    # test on the csv for the last year
    df = pd.read_csv(settings.data_original_csv)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')  
    df = df[df.tourney_date.dt.year == settings.end_year] # test set
    winner_lowest_odds(df)
    winner_highest_rank(df)

baselines()