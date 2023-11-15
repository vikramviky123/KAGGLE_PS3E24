import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from smoker_status.a_constants import *
from smoker_status import logging, CustomException
from smoker_status.b_entity.config_entity import DataTransformationConfig
from smoker_status.f_utils.common import read_yaml


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        try:
            RANDOM_STATE = read_yaml(
                GLOBALPARAMS_FILE_PATH).globalparams.RANDOM_STATE
            df_syn = pd.read_csv(self.config.data_path_syn)
            df_syn.drop('id', axis=1, inplace=True)
            df_act = pd.read_csv(self.config.data_path_act)

            # Merging both datasets
            df = pd.concat([df_syn, df_act], axis=0)
            df.reset_index(drop=True, inplace=True)

            # Dropping the duplicates & restting index
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(
                df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

            train.to_csv(os.path.join(
                self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(
                self.config.root_dir, "test.csv"), index=False)

            logging.info("Splited data into training and test sets")
            logging.info(
                f" Train Shape ==> {train.shape} | test Shape ==> {test.shape}")
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
