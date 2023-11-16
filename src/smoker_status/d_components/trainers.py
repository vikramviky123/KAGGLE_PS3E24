import os

import copy

import pandas as pd
import numpy as np
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score

import optuna
from math import sqrt

import joblib
import pickle

from typing import List, Dict, Set


from smoker_status.a_constants import *
from smoker_status.f_utils.common import read_yaml
from smoker_status.f_utils.common import save_pickle, load_pickle

gparams = read_yaml(GLOBALPARAMS_FILE_PATH).globalparams
RANDOM_STATE = gparams.RANDOM_STATE
N_SPLITS = gparams.N_SPLITS
N_TRIALS = gparams.N_TRIALS


# Set the logging level to ERROR
optuna.logging.set_verbosity(optuna.logging.ERROR)


def eval_metrics(y_true, y_pred, y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    prc = precision_score(y_true, y_pred, zero_division=1)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    return acc, prc, rec, f1, roc

# Load your dataset and define X and y as mentioned in the previous example.


def kfold_cv(X, y, model, metric=roc_auc_score, nsplits=N_SPLITS):
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    trained_models = []
    # oof_df = pd.DataFrame()

    for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # different instance will be created for to dump in pickle/joblib
        # Else same instance will appended to list, which gives same result when using pickle/joblib
        model = copy.deepcopy(model)

        model.fit(X_train, y_train)
        trained_models.append(model)

        y_pred = model.predict_proba(X_val)[:, 1]
        # oof_df['split'+str(idx)] = y_pred
        metric_val = metric(y_val, y_pred)
        metrics.append(metric_val)
        # Validating appended models
        # print(metric_val, metric(y_test,trained_models[idx].predict(X_test)))

    return np.mean(metrics), trained_models
# Load your dataset and define X and y as mentioned in the previous example.


def optimize_hyperparameters(X, y, estimator, hyperparameters, metric_name=roc_auc_score,
                             nsplits=N_SPLITS, ntrials=N_TRIALS):
    def objective(trial):

        params = {}
        for param_name, param_config in hyperparameters.items():
            param_type = param_config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'], step=param_config.get('step', 1))

            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=True)

            elif param_type == 'categorical':
                choices = param_config['choices']
                params[param_name] = trial.suggest_categorical(
                    param_name, choices)

        model = estimator(**params, random_state=RANDOM_STATE)

        # Use K-Fold cross-validation with 5 splits and minimize RMSE
        loss_val = kfold_cv(
            X, y,  model, metric=metric_name, nsplits=nsplits)[0]
        return loss_val

    # We want to minimize RMSE
    study = optuna.create_study(
        direction='maximize', study_name='model_tuning')
    # Adjust the number of trials as needed
    study.optimize(objective, n_trials=ntrials)

    # Get the best hyperparameters
    best_params = study.best_params

    return best_params


def train_kfold_cv(X, y, model, nsplits=N_SPLITS, eval_=None, xtest=None, ytest=None):
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=RANDOM_STATE)

    trained_models = []
    model_score = {}

    model_score['valid'] = {'acc': [], 'prc': [],
                            'rec': [], 'f1': [], 'roc': []}
    model_score['test'] = {'acc': [], 'prc': [],
                           'rec': [], 'f1': [], 'roc': []}

    for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # different instance will be created for to dump in pickle/joblib
        # Else same instance will appended to list, which gives same result when using pickle/joblib
        model = copy.deepcopy(model)

        model.fit(X_train, y_train)
        trained_models.append(model)

        y_pred = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        # Validating appended models
        # print(metric_val, metric(y_test,trained_models[idx].predict(X_test)))

        if eval_ is not None:
            val_metrics = eval_metrics(y_val, y_pred, y_pred_prob)
            model_score['valid']['acc'].append(val_metrics[0])
            model_score['valid']['prc'].append(val_metrics[1])
            model_score['valid']['rec'].append(val_metrics[2])
            model_score['valid']['f1'].append(val_metrics[3])
            model_score['valid']['roc'].append(val_metrics[4])
            if xtest is not None:
                y_pred_test = model.predict(xtest)
                y_pred_prob_test = model.predict_proba(xtest)[:, 1]
                test_metrics = eval_metrics(
                    ytest, y_pred_test, y_pred_prob_test)
                model_score['test']['acc'].append(test_metrics[0])
                model_score['test']['prc'].append(test_metrics[1])
                model_score['test']['rec'].append(test_metrics[2])
                model_score['test']['f1'].append(test_metrics[3])
                model_score['test']['roc'].append(test_metrics[4])

    return trained_models, model_score
