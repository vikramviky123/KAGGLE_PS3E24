import os
import sys
from pathlib import Path

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from smoker_status.a_constants import *
from smoker_status.b_entity.config_entity import ModelEvalConfig
from smoker_status.c_config.configuration import ConfigurationManager
from smoker_status.f_utils.common import save_pickle, load_pickle, read_yaml


class ModelEvaluation:
    def __init__(self, config: ModelEvalConfig):
        self.eval_config = config

    def read_params(self):
        params_yaml = read_yaml(Path(BESTPARAMS_FILE_PATH))
        return params_yaml

    def read_eval_results(self):
        loaded_results = load_pickle(self.eval_config.eval_results)
        return loaded_results

    def log_to_mlflow(self):
        params_yaml = self.read_params()
        loaded_results = self.read_eval_results()

        # remote_server_uri = "https://dagshub.com/vikramviky123/KAGGLE_PS3E24.mlflow"
        mlflow.set_registry_uri(self.eval_config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        for model_name, model_metrics in loaded_results.items():
            # Replace with your actual experiment name
            experiment_name = model_name + "_best_model"

            # Set the experiment name using mlflow.set_experiment
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():

                # Assuming 'test' metrics are used for evaluation, change as needed
                acc = np.mean(model_metrics['test']['acc'])
                prc = np.mean(model_metrics['test']['prc'])
                rec = np.mean(model_metrics['test']['rec'])
                f1 = np.mean(model_metrics['test']['f1'])
                roc = np.mean(model_metrics['test']['roc'])

                # Log best parameters to MLflow
                mlflow.log_params(params_yaml[model_name])

                # Log metrics to MLflow
                mlflow.log_metric("acc", acc)
                mlflow.log_metric("prc", prc)
                mlflow.log_metric("rec", rec)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc", roc)

                # Set the experiment name as a tag (you can also use mlflow.set_tag)
                mlflow.set_tag("experiment_name", experiment_name)

                # Model registry does not work with file store
                # Set tracking_url_type_store based on your configuration
                tracking_url_type_store = "file"

                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.sklearn.log_model(
                        None, model_name, registered_model_name=f"{model_name}_model")
                else:
                    mlflow.sklearn.log_model(None, model_name)
