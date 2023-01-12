import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,create_dir,save_model
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def train(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    train_data = config['artifacts']['train']

    train_path = os.path.join(artifacts_dir,split_data_dir,train_data)
    train_df = pd.read_csv(train_path)
    x = train_df.drop('Outcome',axis=1)
    y = train_df['Outcome']

    XGB = xgb.XGBClassifier()
    RF = RandomForestClassifier()
    DC = DecisionTreeClassifier()
    AD = AdaBoostClassifier()

    CLASS = VotingClassifier(estimators=[('XGB', XGB),('RF',RF),('DC',DC),('AD',AD)])

    

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('CLASS', CLASS)

    ])
    kfold = StratifiedKFold(n_splits=5)
    results = cross_val_score(pipe,x,y, cv=kfold)
    print('Accuracy on train: ',results.mean())
    ensemble_model = pipe.fit(x,y)

    model_dir = config['artifacts']['model']['model_dir']
    model_file = config['artifacts']['model']['model_file']
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    create_dir([model_dir_path])
    model_file_path = os.path.join(model_dir_path,model_file)
    save_model(ensemble_model,model_file_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_arg = args.parse_args()
    train(config_path=parsed_arg.config, params_path=parsed_arg.params)
