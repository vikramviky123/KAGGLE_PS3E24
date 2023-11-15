import os
import sys

from smoker_status import logging, CustomException

from smoker_status.a_constants import *
from smoker_status.b_entity.config_entity import (DataIngestionConfig,
                                                  DataTransformationConfig,
                                                  ModelTrainerConfig,
                                                  ModelEvalConfig)
from smoker_status.f_utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir,
                            config.downloaded_dir,
                            config.extracted_dir])

        data_ingestion_config = DataIngestionConfig(root_dir=config.root_dir,
                                                    syn_URL=config.syn_URL,
                                                    act_URL=config.act_URL,
                                                    downloaded_dir=config.downloaded_dir,
                                                    extracted_dir=config.extracted_dir,
                                                    file_path_syn=config.file_path_syn,
                                                    file_path_act=config.file_path_act)

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path_syn=config.data_path_syn,
            data_path_act=config.data_path_act
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            target=config.target,
        )

        return model_trainer_config

    def get_model_eval_config(self) -> ModelEvalConfig:
        config = self.config.model_eval

        create_directories([config.root_dir])

        model_eval_config = ModelEvalConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            eval_results=Path(config.eval_results),
            best_params=Path(config.best_params),
            mlflow_uri=config.mlflow_uri
        )

        return model_eval_config
