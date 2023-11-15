from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    syn_URL: str
    act_URL: str
    downloaded_dir: Path
    extracted_dir: Path
    file_path_syn: Path
    file_path_act: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path_syn: Path
    data_path_act: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target: str


@dataclass(frozen=True)
class ModelEvalConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    eval_results: Path
    best_params: Path
    mlflow_uri: str
