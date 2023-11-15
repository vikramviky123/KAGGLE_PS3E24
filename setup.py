from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


with open("README.md", 'r', encoding="utf-8") as f:
    long_description = f.read()

SRC_REPO = "smoker"
__VERSION__ = "0.0.0"

AUTHOR_USER_NAME = "vikramviky123"
AUTHOR_EMAIL = "vikram_viky2001@yahoo.com"

REPO_NAME = "KAGGLE_PS3E14"


setup(
    name=SRC_REPO,
    version=__VERSION__,

    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,

    description="Python Package for SMoker prediction",
    long_description=long_description,

    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"},

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=get_requirements('requirements.txt')

)
