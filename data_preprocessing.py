import os
import pandas as pd

import settings



# helper function to get the initial dataframe from an excel file
def get_df_from_excel(xl_name):
    xls = f'{xl_name}.xls'
    xlsx = f'{xl_name}.xlsx'
    if os.path.isfile(xls):
        return pd.read_excel(xls)
    elif os.path.isfile(xlsx):
        return pd.read_excel(xlsx)
    else:
        print(f'Could not find file {xl_name}')

# function to merge data from the two sources
def merge_dfs(df1, df2, out_file):
    merged = 0
    winner_odds_col, loser_odds_col, retirement_col = [], [], []

    for i in range(len(df2)):
        matches = df1[(df1.WPts == df2.loc[i, 'winner_rank_points'] ) 
                      & (df1.LPts == df2.loc[i, 'loser_rank_points'])
                      & (df1.WRank == df2.loc[i, 'winner_rank']) 
                      & (df1.LRank == df2.loc[i, 'loser_rank'])]
        winner_odds, loser_odds, retirement = 0, 0, 0
        if len(matches) == 1:
            m = matches.iloc[0]
            if not pd.isna(m.B365W) and not pd.isna(m.B365L):
                winner_odds, loser_odds = m.B365W, m.B365L
                merged += 1
                retirement = 0 if m.Comment == 'Completed' else 1
        winner_odds_col.append(winner_odds)
        loser_odds_col.append(loser_odds)
        retirement_col.append(retirement)

    df2['retirement'] = retirement_col
    df2['winner_odds'] = winner_odds_col
    df2['loser_odds'] = loser_odds_col
    total = len(df2)
    df2 = df2[(df2['winner_odds'] != 0) & (df2['loser_odds'] != 0)]
    df2.to_csv(out_file, index=False)
    return(merged, total)


def merge_files_yearly():
    print('Merging yearly files')
    merged, total = 0, 0
    for year in range(settings.start_year, settings.end_year+1):
        dfy1 = get_df_from_excel(f'{settings.path_td}/{year}')
        dfy2 = pd.read_csv(f'{settings.path_js}/atp_matches_{year}.csv')
        out_file = f'{settings.path_merged}/{year}.csv'
        (merged_y, total_y) = merge_dfs(dfy1, dfy2, out_file)
        print(f'Merged {merged_y} of the {total_y} rows for year {year}.')
        merged += merged_y
        total += total_y
    print(f'\nMerged {merged} of the total {total} rows ({merged/total*100}%)')
    return (merged, total)

def preprocess():
    (merged, total) = merge_files_yearly()
    # get the df for the first file
    df = pd.read_csv(f'{settings.path_merged}/{settings.start_year}.csv')

    # concat the df of the next files until the final year
    for year in range(settings.start_year+1, settings.end_year+1):
        dfy = pd.read_csv(f'{settings.path_merged}/{year}.csv')
        df = pd.concat([df, dfy], ignore_index=True)

    n_rows = len(df)
    if n_rows != merged:
        print(f'WARNING. The dataframe derived from merging the files contains {n_rows} of the {merged} original rows')

    # simple cleaning to take out rows where the bet365 odds are absent
    df.dropna(subset=['winner_ht', 'loser_ht', 'winner_age', 'loser_age'], inplace=True) 
    print(f'\nAfter cleaning, there are {len(df)} of the {n_rows} rows ({len(df)/n_rows}%, {len(df)/total}% of the total {total})')

    df.to_csv(settings.data_original_csv)
    print(f'Saved data into {settings.data_original_csv}')


preprocess()
