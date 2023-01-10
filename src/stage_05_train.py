import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,create_dir,save_model
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline



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

    sd = StandardScaler()

    criterion = params['train']['criterion']
    max_depth = params['train']['max_depth']
    n_estimators = params['train']['n_estimators']
    min_samples_split = params['train']['min_samples_split']
    

    randomforestclassifier = RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split, n_estimators=n_estimators)

    pipe = imbpipeline([('smt', SMOTE()),('sd',sd), ('randomforestclassifier',randomforestclassifier )])

    pipe.fit(x,y)

    model_dir = config['artifacts']['model']['model_dir']
    model_file = config['artifacts']['model']['model_file']
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    create_dir([model_dir_path])
    model_file_path = os.path.join(model_dir_path,model_file)
    save_model(pipe,model_file_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_arg = args.parse_args()
    train(config_path=parsed_arg.config, params_path=parsed_arg.params)
