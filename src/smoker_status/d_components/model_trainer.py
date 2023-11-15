import os
import sys
from pathlib import Path
import yaml

from smoker_status import logging, CustomException
from smoker_status.a_constants import *
from smoker_status.f_utils.common import read_yaml
from smoker_status.f_utils.common import save_pickle, load_pickle

from smoker_status.b_entity.config_entity import ModelTrainerConfig
from smoker_status.c_config.configuration import ConfigurationManager
from smoker_status.d_components.trainers import (kfold_cv,
                                                 train_kfold_cv,
                                                 optimize_hyperparameters,
                                                 roc_auc_score,
                                                 eval_metrics)

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.gparams = read_yaml(GLOBALPARAMS_FILE_PATH).globalparams
        self.params = read_yaml(PARAMS_FILE_PATH)

        self.train_models_path = Path(self.config.root_dir)
        self.train_path = Path(self.config.train_data_path)
        self.test_path = Path(self.config.test_data_path)
        self.model_name = self.config.model_name
        self.target = self.config.target

        self.RANDOM_STATE = self.gparams.RANDOM_STATE
        self.N_SPLITS = self.gparams.N_SPLITS
        self.N_TRIALS = self.gparams.N_TRIALS

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

    def split_data(self):
        train = self.load_data()[0]
        test = self.load_data()[1]

        y_col = self.target
        x_cols = [col for col in train.columns if col not in [y_col]]
        X = train[x_cols]
        y = train[y_col]
        xtest = test[x_cols]
        ytest = test[y_col]
        return X, y, xtest, ytest

    def train_models(self):
        X, y, xtest, ytest = self.split_data()

        model_dict = {'lgbm_classifier': lgb.LGBMClassifier,
                      'xgb_classifier': xgb.XGBClassifier,
                      'histgradient_boost_classifier': HistGradientBoostingClassifier,
                      'random_forest_classifier': RandomForestClassifier
                      }

        params_dict = {}
        trained_models = {}
        evaluation_metrics = {}

        for model_name, params in self.params.items():
            logging.info(f"tuning parameters for model: {model_name}")

            # Assuming model_dict contains your model instances
            model_ = model_dict[model_name]
            best_params = optimize_hyperparameters(X, y,
                                                   model_,
                                                   params,
                                                   metric_name=roc_auc_score,
                                                   nsplits=self.N_SPLITS, ntrials=self.N_TRIALS)

            params_dict[model_name] = best_params
            logging.info(f"tuning done for == {model_name}== DONE")
            logging.info(f"-"*80)

        for model_name, best_params in params_dict.items():
            logging.info(f"Initiating model : {model_name}")
            model = model_dict[model_name]
            best_model = model(**best_params, random_state=self.RANDOM_STATE)

            trained_models[model_name], evaluation_metrics[model_name] = train_kfold_cv(
                X, y,
                best_model,
                nsplits=self.N_SPLITS,
                eval_='yes',
                xtest=xtest, ytest=ytest)

            logging.info(
                f"Model =={model_name}== trained with best params == DONE")
            logging.info(f"-"*80)

        return trained_models, params_dict, evaluation_metrics

    def save_models(self, trained_models, params_dict, evaluation_metrics):
        joblib_file_path = Path(os.path.join(
            self.train_models_path, self.model_name))
        joblib_best_params = Path(os.path.join(
            self.train_models_path, 'best_params.joblib'))
        joblib_eval_results = Path(os.path.join(
            self.train_models_path, 'eval_results.joblib'))

        with open('bestparams.yaml', 'w') as yaml_file:
            yaml.dump(params_dict, yaml_file, default_flow_style=False)

        logging.info(f"best parameters SAVED as bestparams.yaml")

        save_pickle(trained_models, joblib_file_path)
        logging.info(f"trained models are SAVED in {joblib_file_path}")
        save_pickle(params_dict, joblib_best_params)
        logging.info(f"best params dict SAVED in {joblib_best_params}")
        save_pickle(evaluation_metrics, joblib_eval_results)
        logging.info(
            f"evaluation metrics of valid and test SAVED in {joblib_eval_results}")
