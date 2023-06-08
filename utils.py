import pandas as pd

def get_sorted_odds(df):
  winner_odds = [row['p0_odds'] if row['winner'] == 0 else row['p1_odds'] for index, row in df.iterrows()]
  loser_odds = [row['p1_odds'] if row['winner'] == 0 else row['p0_odds'] for index, row in df.iterrows()] 
  return (winner_odds, loser_odds)

"""
def get_x_y_w(csv_file):
    df = pd.read_csv(csv_file)
    (weights, _) = get_sorted_odds(df)
    y = df.pop('winner')
    df.pop('p1_odds')
    X = df
    return (X, y, weights)
"""

def get_x_y_w(csv_file):
    df = pd.read_csv(csv_file)
    (weights, _) = get_sorted_odds(df)
    y = df.pop('winner')
    df.pop('p1_odds')
    df.pop('p0_odds')
    X = df
    return (X, y, weights)