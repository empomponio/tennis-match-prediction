import os, random
import numpy as np
import tensorflow as tf
import settings

from data_preprocessing import preprocess
from feature_extraction import extract_features
from baselines import evaluate_baselines
from model_tuning import hypertune_all
from feature_analysis import analyze_features
from model_evaluation import evaluate_cv, evaluate_test, evaluate_xgb_fscore



def set_random_seeds():
    os.environ['PYTHONHASHSEED']=str(settings.rnd_seed)
    tf.random.set_seed(settings.rnd_seed)
    np.random.seed(settings.rnd_seed)
    random.seed(settings.rnd_seed)


# 0. Fix seeds for reproducibility
set_random_seeds()

# 1. Merge original yearly dataset into a single file
#preprocess()

# 2. Create custom features and generate ML-ready feature files
#extract_features()

# 3. Evaluate baseline and challeger models
#evaluate_baselines()

# 4. Find the best hyperparameters on all ML models
#hypertune_all()

# 5. Analyze features in tree-base models
#analyze_features()

# 6. Evaluate best models on cross-validation
#evaluate_cv()

# 7. Evaluate final model on test set 
#evaluate_test()

# 8. Experiments on xgb tuned on F-score (beta=0.5)
evaluate_xgb_fscore()