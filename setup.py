
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "diabetes_predication_dvc"
AUTHOR_USER_NAME = "hardik desai"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['dvc','pandas','scikit-learn']


setup(
    name='src',
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for DVC- usecase",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="hardikdesai262@gmail.com",
    packages=[SRC_REPO],
   
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)