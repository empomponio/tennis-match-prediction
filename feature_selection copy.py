from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

import settings


def get_hist_cols():
    hist_vars = ['wins', 'played', 'winperc']
    hist_cols = []
    hist_cols += [col + '_year' for col in hist_vars]
    hist_cols += [col + '_month' for col in hist_vars]
    hist_cols += [col + '_surface' for col in hist_vars]
    hist_cols += [col + '_level' for col in hist_vars]
    hist_cols += ['winperc_tourney', 'winperc_matchup', 'rank_gain', 'rank_points_gain', 'momentum']
    return hist_cols

# gets all columns of the dataframe containing the data features of both winner and loser
def get_all_data_cols():
    p_basic_cols = ['name', 'height', 'age', 'rank', 'rank_points']
    basic_cols = ['w_' + col for col in p_basic_cols]
    basic_cols += ['l_' + col for col in p_basic_cols]

    p_hist_cols = get_hist_cols()
    hist_cols = ['w_' + col for col in p_hist_cols]
    hist_cols += ['l_' + col for col in p_hist_cols]

    p_details_cols = [
            'servegames_perc', 'returngames_perc', 
            'won_1stserve_perc', 'won_2ndserve_perc', 'won_1streturn_perc', 'won_2ndreturn_perc', 
            'bp_saved_perc', 'match_duration'
            ]
    details_cols = ['w_' + col for col in p_details_cols]
    details_cols += ['l_' + col for col in p_details_cols]

    p_cat_cols = ['hand', 'wildcard']
    cat_cols = ['w_' + col for col in p_cat_cols]
    cat_cols += ['l_' + col for col in p_cat_cols]    

    odds_cols = ['w_odds', 'l_odds']

    return basic_cols + hist_cols + details_cols + cat_cols + odds_cols

# function that takes as input the original df and concatenates it with the data for start_year-1
# we need to add the that in order to get the part features (history and details) for start_year 
def get_augmented_df(df):
    # data for year preceding the starting one
    df_prev = pd.read_csv(f'{settings.path_js}/atp_matches_{settings.start_year-1}.csv') # need the [columns] to preserve the order of the columns that we want
    df_prev['tourney_date'] = pd.to_datetime(df_prev['tourney_date'], format='%Y%m%d')
    df_concat = pd.concat([df_prev, df], ignore_index=True) # augmented dataframe containing data from start_year-1
    return df_concat

def get_previous_results(past_matches, player, tourney_date, match_num, time_window=0, surface='', tourney_level='', tourney_name='', opponent=''):
    if time_window != 0:
        past_matches = past_matches[ 
                ((tourney_date-past_matches.tourney_date) / np.timedelta64(1, 'D') <= time_window)            # include matches within the chosen number of days
        ]

    if surface != '':
        past_matches = past_matches[past_matches.surface == surface]
    if tourney_level != '':
        past_matches = past_matches[past_matches.tourney_level == tourney_level]
    if tourney_name != '':
        past_matches = past_matches[past_matches.tourney_name == tourney_name]    
    if opponent != '':
        past_matches = past_matches[(past_matches.winner_name == opponent) | (past_matches.loser_name == opponent)]
    wins_rows = past_matches[(past_matches.winner_name == player)]
    losses_rows = past_matches[(past_matches.loser_name == player)]

    wins, losses = len(wins_rows), len(losses_rows) 
    win_perc = wins / (wins+losses) if wins != 0 else 0
    return (wins, wins+losses, win_perc)

def get_details_past_matches(past_matches, player, tourney_date):
    won_matches = past_matches[past_matches.winner_name == player]
    lost_matches = past_matches[past_matches.loser_name == player]

    total_service_pts = won_matches['w_svpt'].sum() + lost_matches['l_svpt'].sum()
    won_service_pts = won_matches['w_1stWon'].sum() + won_matches['w_2ndWon'].sum() + lost_matches['l_1stWon'].sum() + lost_matches['l_2ndWon'].sum()
    serve_games_perc = won_service_pts / total_service_pts if total_service_pts != 0 else 0

    total_service_pts_opp = won_matches['l_svpt'].sum() + lost_matches['w_svpt'].sum()
    won_service_pts_opp = won_matches['l_1stWon'].sum() + won_matches['l_2ndWon'].sum() + lost_matches['w_1stWon'].sum() + lost_matches['w_2ndWon'].sum()
    return_games_perc = 1 - (won_service_pts_opp / total_service_pts_opp)    if total_service_pts_opp != 0 else 0

    total_bp_saved = won_matches['w_bpSaved'].sum() + lost_matches['l_bpSaved'].sum()
    total_bp_faced = won_matches['w_bpFaced'].sum() + lost_matches['l_bpFaced'].sum()
    bp_save_perc = total_bp_saved / total_bp_faced if total_bp_faced != 0 else 0

    tourney_matches = won_matches[(won_matches.tourney_date == tourney_date)]
    total_duration = tourney_matches['minutes'].sum()
    num_matches = len(tourney_matches)
    match_dur = total_duration / num_matches if num_matches != 0 else 0

    total_1st_in = won_matches['w_1stIn'].sum()+lost_matches['l_1stIn'].sum()
    won_1st = won_matches['w_1stWon'].sum()+lost_matches['l_1stWon'].sum()
    won_1st_perc = won_1st / total_1st_in if total_1st_in != 0 else 0
    total_2nd_points = total_service_pts - total_1st_in
    won_2nd = won_matches['w_2ndWon'].sum()+lost_matches['l_2ndWon'].sum()
    won_2nd_perc = won_2nd / total_2nd_points if total_2nd_points > 0 else 0

    total_1st_in_opp = won_matches['l_1stIn'].sum()+lost_matches['w_1stIn'].sum()
    won_1st_opp = won_matches['l_1stWon'].sum()+lost_matches['w_1stWon'].sum()
    won_1st_perc_opp = won_1st_opp / total_1st_in_opp if total_1st_in_opp != 0 else 0
    won_1st_ret_perc = 1 - won_1st_perc_opp if won_1st_perc_opp != 0 else 0
    
    total_2nd_points_opp = total_service_pts_opp - total_1st_in_opp
    won_2nd_opp = won_matches['l_2ndWon'].sum()+lost_matches['w_2ndWon'].sum()
    won_2nd_perc_opp = won_2nd_opp / total_2nd_points_opp if total_2nd_points_opp != 0 else 0
    won_2nd_ret_perc = 1 - won_2nd_perc_opp if won_2nd_perc_opp else 0
    
    return [serve_games_perc, return_games_perc, won_1st_perc, won_2nd_perc, won_1st_ret_perc, won_2nd_ret_perc, bp_save_perc, match_dur]


def get_rank_gains(df, tourney_date, player, rank, rank_points):
    last_matches = df[(tourney_date > df.tourney_date) 
        & ((df.winner_name == player) | (df.loser_name == player))]

    if len(last_matches) == 0:
        return 0, 0
    last_match = last_matches.iloc[last_matches.tourney_date.argmax()]

    last_rank_points = last_match.winner_rank_points if last_match.winner_name == player else last_match.loser_rank_points
    last_rank = last_match.winner_rank if last_match.winner_name == player else last_match.loser_rank
    rank_gain = last_rank - rank
    days_diff = (tourney_date-last_match.tourney_date) / np.timedelta64(1, 'D')
    rank_points_gain = rank_points - last_rank_points
    rank_gain /= days_diff
    rank_points_gain /= days_diff
    return (rank_gain, rank_points_gain)

def get_momentum_data(df, tourney_date, match_num, player):
    tourney_matches = df[(tourney_date == df.tourney_date) & (df.match_num < match_num) & (df.winner_name == player)]
    momentum = 1
    for i in range(len(tourney_matches)):
        momentum *= tourney_matches.iloc[i].winner_odds
    return momentum


def make_features_csv():
    print('Creating data features....')

    # get columns for df
    columns = get_all_data_cols()
    columns += ['tourney_date']
    # get original df
    df = pd.read_csv(settings.data_original_csv)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    # get augmented df
    df_concat = get_augmented_df(df)

    data = []
    # for each row in the original df:
    for i in range(len(df)):
        row = []
        dp = df.loc[i]

        basic_features = [
                dp.winner_name, dp.winner_ht, dp.winner_age, dp.winner_rank, dp.winner_rank_points,
                dp.loser_name, dp.loser_ht, dp.loser_age, dp.loser_rank, dp.loser_rank_points
                ]
        row += basic_features

        # init variables for past features
        tourney_date = dp.tourney_date
        match_num = dp.match_num
        surface = dp.surface
        tourney_level = dp.tourney_level
        tourney_name = dp.tourney_name

        past_matches = df_concat[
                (tourney_date >= df_concat.tourney_date)                                                                                            # include only past matches
                & ~ ((tourney_date == df_concat.tourney_date) & (df_concat.match_num >= match_num))     # exclude the match itself and the following matches in the same tournament
                ] 

        hist_features = []
        det_features = []
        for player in ['winner', 'loser']:
            player_name = dp[player+'_name']
            opponent = 'loser' if player == 'winner' else 'winner'
            opponent_name = dp[opponent+'_name']
            # get historical features: wins, played matches, win percentages
            (wins_y, played_y, winperc_y) = get_previous_results(past_matches, player_name, tourney_date, match_num, time_window=365)
            (wins_m, played_m, winperc_m) = get_previous_results(past_matches, player_name, tourney_date, match_num, time_window=30)
            (wins_surf, played_surf, winperc_surf) = get_previous_results(past_matches, player_name, tourney_date, match_num, time_window=365, surface=surface)
            (wins_lev, played_lev, winperc_lev) = get_previous_results(past_matches, player_name, tourney_date, match_num, time_window=365, tourney_level=tourney_level)
            (_, _, winperc_tourney) = get_previous_results(past_matches, player_name, tourney_date, match_num, time_window=400, tourney_name=tourney_name)
            (_, _, winperc_matchup) = get_previous_results(past_matches, player_name, tourney_date, match_num, opponent=opponent_name)


            hist_features += [
                    wins_y, played_y, winperc_y, 
                    wins_m, played_m, winperc_m,
                    wins_surf, played_surf, winperc_surf,
                    wins_lev, played_lev, winperc_lev,
                    winperc_tourney,
                    winperc_matchup
                    ]
            
            # get data on rank gains
            rank = dp[player+'_rank']
            rank_points = dp[player+'_rank_points']
            (rank_gain, rank_points_gain)= get_rank_gains(df, tourney_date, player_name, rank, rank_points)
            # get data on momentum in tournament
            momentum = get_momentum_data(df, tourney_date, match_num, player_name)
            
            hist_features += [rank_gain, rank_points_gain, momentum]

            # get details features 
            time_window = 365 # include matches within the chosen number of days
            past_matches_tw = past_matches[((tourney_date-past_matches.tourney_date) / np.timedelta64(1, 'D') <= time_window)]
            det_features += get_details_past_matches(past_matches_tw, player_name, tourney_date)

        row += hist_features + det_features

        # get categorical features
        winner_hand = 'U' if dp.loser_hand == 'U' else dp.winner_hand
        loser_hand = 'U' if dp.winner_hand == 'U' else dp.loser_hand
        winner_wildcard = 'Y' if dp.winner_entry == 'WC' else 'N'
        loser_wildcard = 'Y' if dp.loser_entry == 'WC' else 'N'
        cat_features = [winner_hand, winner_wildcard, loser_hand, loser_wildcard]
        row += cat_features
        row += [dp.winner_odds, dp.loser_odds]

        row += [dp.tourney_date]
        data.append(row)
        if i==0 or (i+1)%1000 == 0 or i+1==len(df):
            print(f'Computed details for {i+1}/{len(df)} matches.')
    
    # create df with columns and data
    df_features = pd.DataFrame(data=data, columns=columns)
    df_features.to_csv(settings.data_csv, index=False)
    print(f'Saved data into {settings.data_csv}')


# gets all columns of the dataframe containing the merged features
def get_all_features_cols():
    basic_cols = ['rank_diff', 'rank_logdiff', 'rank_points_diff', 'rank_points_logdiff', 'height_diff', 'age_diff']

    hist_vars =    ['wins_diff', 'played_diff', 'winperc_diff']
    hist_cols = []
    hist_cols += [col + '_year' for col in hist_vars]
    hist_cols += [col + '_month' for col in hist_vars]
    hist_cols += [col + '_surface' for col in hist_vars]
    hist_cols += [col + '_level' for col in hist_vars]
    hist_cols += ['winpercdiff_tourney', 'winpercdiff_matchup', 'rankgain_diff', 'rankpointsgain_diff', 'momentum_diff']

    det_cols = [
            'serveperc_diff', 'returnperc_diff', 
            'serve1perc_diff', 'return1perc_diff', 'serve2perc_diff', 'return2perc_diff', 
            'bpsavedperc_pdiff', 'matchduration_diff', 'servereturn_diff',
            'servereturn1_diff', 'servereturn2_diff'
    ]

    cat_cols = ['hand_matchup', 'wildcard']    # these are the columns *before* one-hot encoding
    
    return (basic_cols, hist_cols, det_cols, cat_cols)



def get_basic_features(dp):
    rank_diff = dp.p1_rank - dp.p2_rank
    # compute logarithmic difference between ranks (can never be 0 because ranks are different even when rank points are the same)
    rank_logdiff = np.log(rank_diff) if rank_diff > 0 else -np.log(-rank_diff) 
    rank_points_diff = dp.p1_rank_points - dp.p2_rank_points
    # compute logarithmic difference between rank points (the logdiff will be set to 0 when rank points are equal (otherwise it would be infinite))
    rank_points_logdiff = np.log(rank_points_diff) if rank_points_diff > 0 else 0 if rank_points_diff == 0 else -np.log(-rank_points_diff) 
    height_diff = dp.p1_height - dp.p2_height 
    age_diff = dp.p1_age - dp.p2_age
    return [rank_diff, rank_logdiff, rank_points_diff, rank_points_logdiff, height_diff, age_diff]


def get_hist_features(dp):
    wins_diff_y = dp.p1_wins_year - dp.p2_wins_year
    played_diff_y = dp.p1_played_year - dp.p2_played_year
    winperc_diff_y = dp.p1_winperc_year - dp.p2_winperc_year

    wins_diff_m = dp.p1_wins_month - dp.p2_wins_month
    played_diff_m = dp.p1_played_month - dp.p2_played_month
    winperc_diff_m = dp.p1_winperc_month - dp.p2_winperc_month

    wins_diff_surf = dp.p1_wins_surface - dp.p2_wins_surface
    played_diff_surf = dp.p1_played_surface - dp.p2_played_surface
    winperc_diff_surf = dp.p1_winperc_surface - dp.p2_winperc_surface

    wins_diff_lev = dp.p1_wins_level - dp.p2_wins_level
    played_diff_lev = dp.p1_played_level - dp.p2_played_level
    winperc_diff_lev = dp.p1_winperc_level - dp.p2_winperc_level

    winpercdiff_tourney = dp.p1_winperc_tourney - dp.p2_winperc_tourney
    winpercdiff_matchup = dp.p1_winperc_matchup - dp.p2_winperc_matchup

    rankgain_diff = dp.p1_rank_gain - dp.p2_rank_gain
    rankpointsgain_diff = dp.p1_rank_points_gain - dp.p2_rank_points_gain
    momentum_diff = dp.p1_momentum - dp.p2_momentum

    return [
            wins_diff_y, played_diff_y, winperc_diff_y,
            wins_diff_m, played_diff_m, winperc_diff_m,
            wins_diff_surf, played_diff_surf, winperc_diff_surf,
            wins_diff_lev, played_diff_lev, winperc_diff_lev,
            winpercdiff_tourney, winpercdiff_matchup,
            rankgain_diff, rankpointsgain_diff, momentum_diff
    ]

def get_det_features(dp):  
    serveperc_diff = dp.p1_servegames_perc - dp.p2_servegames_perc 
    returnperc_diff = dp.p1_returngames_perc - dp.p2_returngames_perc 
    serve1perc_diff = dp.p1_won_1stserve_perc - dp.p2_won_1stserve_perc
    return1perc_diff = dp.p1_won_1streturn_perc - dp.p2_won_1streturn_perc
    serve2perc_diff = dp.p1_won_2ndserve_perc - dp.p2_won_2ndserve_perc
    bpsavedperc_diff = dp.p1_bp_saved_perc - dp.p2_bp_saved_perc
    return2perc_diff = dp.p1_won_2ndreturn_perc - dp.p2_won_2ndreturn_perc
    matchdur_diff = dp.p1_match_duration - dp.p2_match_duration

    # comparing how strong player1's attacking game is vs player2's defending game and viceversa
    servereturn_diff = (dp.p1_servegames_perc - dp.p2_returngames_perc) - (dp.p2_servegames_perc - dp.p1_returngames_perc)
    servereturn1_diff = (dp.p1_won_1stserve_perc - dp.p2_won_1streturn_perc) - (dp.p2_won_1stserve_perc - dp.p1_won_1streturn_perc)
    servereturn2_diff = (dp.p1_won_2ndserve_perc - dp.p2_won_2ndreturn_perc) - (dp.p2_won_2ndserve_perc - dp.p1_won_2ndreturn_perc)

    return [
            serveperc_diff, returnperc_diff, 
            serve1perc_diff, return1perc_diff, 
            serve2perc_diff, return2perc_diff, 
            bpsavedperc_diff, matchdur_diff,
            servereturn_diff, servereturn1_diff, servereturn2_diff
            ]

def get_cat_features(dp):
        hand_matchup = dp.p1_hand + dp.p2_hand
        wildcard_matchup = dp.p1_wildcard + dp.p2_wildcard 
        return [hand_matchup, wildcard_matchup]


def create_train_test_csv(df, csv_path, csv_path_test):
    # init test dataset with the data from the last year
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df_test = df[df.tourney_date.dt.year == settings.end_year] # test set
    df = df[df.tourney_date.dt.year < settings.end_year] # train/validation set
    print(f'Training/validation dataframe shape: {df.shape}\nTest dataframe shape: {df_test.shape}')
    # remove column for date because it is not used by the ML algorithms
    df.pop('tourney_date')
    df_test.pop('tourney_date')
    # save files
    df.to_csv(csv_path, index=False)
    print('Saved training/validation dataframe in path', csv_path)
    df_test.to_csv(csv_path_test, index=False)
    print('Saved test dataframe in path', csv_path_test)

# creates the dataframe for the ML pipeline and saves it to a csv_file
# features of the two player are merged in one (e.g. with differences or combinations)
def create_csv_merge(df):
    print(f'\n\nMerging features...')
    # derive features: difference in rank, rank points, height, age, and hand combination
    (basic_cols, hist_cols, det_cols, cat_cols) = get_all_features_cols()
    columns = basic_cols + hist_cols + det_cols + cat_cols + ['p0_odds', 'p1_odds', 'winner', 'tourney_date']

    data = []
    for i in range(len(df)):
        dp = df.loc[i]
        basic_features = get_basic_features(dp)
        hist_features = get_hist_features(dp)
        det_features = get_det_features(dp)
        cat_features = get_cat_features(dp)
        # append columns
        data.append(basic_features +
                    hist_features +
                    det_features +
                    cat_features +
                    [dp.p1_odds, dp.p2_odds, dp.winner, dp.tourney_date])

    # create dataframe
    df = pd.DataFrame(columns=columns, data=data)

    # label encoding
    for cat_col in cat_cols:
        df[cat_col] = LabelEncoder().fit_transform(np.array(df[cat_col]).reshape((-1)))

    create_train_test_csv(df, settings.data_features_csv, settings.data_features_test_csv)


def sort_player_data(df):
    print('***********************************************************************************')
    # update names of columns to refer to player1 and player2 instead of winner and loser
    cols = get_all_data_cols()
    updated_cols = ['p1'+col[1:] if col.startswith('w') else 'p2'+col[1:] for col in cols]
    updated_cols += ['tourney_date']
    p1_cols = [col for col in updated_cols if col.startswith('p1')]
    p2_cols = [col for col in updated_cols if col.startswith('p2')]
    df.columns = updated_cols
    # add column for winner
    df['winner'] = 0

    # df columns that will be swapped: player 2 has higher rank (lower rank number) than player 1
    df_swapped = df[df.p1_rank > df.p2_rank]
    print(f'Found {len(df_swapped)} out of {len(df)} columns where player with lower rank won the match. ({len(df_swapped)/len(df)}%)')
    # swap columns for player 1 and 2
    df_swapped[p1_cols+p2_cols] = df_swapped[p2_cols+p1_cols]
    # for these rows, the winner will be player 2
    df_swapped.loc[:, 'winner'] = 1
    df.update(df_swapped)


def feature_selection():
    # step 1: create initial csv containing augmented features
    #make_features_csv()

    # step 2: create csv containing numerical features only
    # features of both players will be merged
    df = pd.read_csv(settings.data_csv)
    sort_player_data(df)
    create_csv_merge(df)

feature_selection()
