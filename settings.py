rnd_seed = 123

path_original = 'data/original'
path_yearly = f'{path_original}/yearly'
data_original_csv = f'{path_original}/data_original.csv'                       # csv containing the full data from the selected years

path_td = f'{path_yearly}/tennis-data.co.uk'
path_js = f'{path_yearly}/JeffSackmann'
path_merged = f'{path_yearly}/merged'
start_year, end_year = 2006, 2022

path_generated = 'data/generated'
data_csv = f'{path_generated}/data.csv'                                                 # csv with updated columns after first part of feature selection
data_features_le_csv = f'{path_generated}/data_features_le_.csv'                               # csv with numerical features only, ready to be used in ML algorithms (training and validation)
data_features_le_test_csv = f'{path_generated}/data_features_le_test.csv'                     # csv with numerical features only, ready to be used in ML algorithms (test)
data_features_onehot_csv = f'{path_generated}/data_features_onehot.csv'                 # csv with numerical features only, ready to be used in ML algorithms (training and validation)
data_features_onehot_test_csv = f'{path_generated}/data_features_onehot_test.csv'       # csv with numerical features only, ready to be used in ML algorithms (test)

path_feature_files = f'{path_generated}/feature_files'
data_symmetric_csv = f'{path_feature_files}/symmetric.csv'
data_symmetric_test_csv = f'{path_feature_files}/symmetric_test.csv'

data_double_csv = f'{path_feature_files}/double.csv'
data_double_test_csv = f'{path_feature_files}/double_test.csv'
data_baseline_test_csv = f'{path_feature_files}/baseline_test.csv'



path_feature_elimination = f'{path_generated}/featureElimination'
data_features_lr_csv = f'{path_feature_elimination}/data_features_lr.csv'               # features are selected with a logistic regression estimator         
data_features_lr_test_csv = f'{path_feature_elimination}/data_features_lr_test.csv'
data_features_svc_csv = f'{path_feature_elimination}/data_features_svc.csv'             # features are selected with a support vector estimator         
data_features_svc_test_csv = f'{path_feature_elimination}/data_features_svc_test.csv' 
data_features_rf_csv = f'{path_feature_elimination}/data_features_rf.csv'               # feature selected according to importance computed with random forest as decrease in impurity         
data_features_rf_test_csv = f'{path_feature_elimination}/data_features_rf_test.csv'
data_features_hp_csv = f'{path_feature_elimination}/data_features.csv'                  # features are hand-picked according to interpretation of previous results        
data_features_hp_test_csv = f'{path_feature_elimination}/data_features_test.csv'


path_evaluation = 'data/evaluation'
data_train_csv = f'{path_evaluation}/data_train.csv'
data_eval_csv = f'{path_evaluation}/data_eval.csv'
path_hypertuning = f'{path_evaluation}/hypertuning'

path_plots = 'plots'
