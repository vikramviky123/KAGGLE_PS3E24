import os
import sys
from pathlib import Path

from smoker_status import logging, CustomException
from smoker_status.b_entity.config_entity import ModelTrainerConfig
from smoker_status.c_config.configuration import ConfigurationManager
from smoker_status.d_components.model_trainer import ModelTrainer

STAGE_NAME = "MODEL -- TRAINING -- STAGE"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        trained_models, params_dict, evaluation_metrics = model_trainer.train_models()
        model_trainer.save_models(
            trained_models, params_dict, evaluation_metrics)


if __name__ == '__main__':
    try:
        logging.info(
            f"\n\nx==========x\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x\n\n")
    except Exception as e:
        logging.exception(CustomException(e, sys))
        raise CustomException(e, sys)
