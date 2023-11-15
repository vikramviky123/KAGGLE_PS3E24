import os
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s:')


project_name = 'smoker_status'


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/a_constants/__init__.py",
    f"src/{project_name}/b_entity/__init__.py",
    f"src/{project_name}/b_entity/config_entity.py",
    f"src/{project_name}/c_config/__init__.py",
    f"src/{project_name}/c_config/configuration.py",
    f"src/{project_name}/d_components/__init__.py",
    f"src/{project_name}/e_pipeline/__init__.py",
    f"src/{project_name}/f_utils/__init__.py",
    f"src/{project_name}/f_utils/common.py",
    'research/trials.ipynb',
    'templates/Intro.html',
    'static/css/style.css',
    'config/config.yaml',
    'dvc.yaml',
    'params.yaml',
    'globalparams.yaml',
    'requirements.txt',
    'setup.py',
    'app.py',
    'main.py'
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Directory =={filedir}== is created")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"File =={filename}== is created")

    else:
        logging.info(f"=={filename}==  already exists!!!!!")
