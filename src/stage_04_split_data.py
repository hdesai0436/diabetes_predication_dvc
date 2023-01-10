import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,create_dir,save_local_df
import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts_dir = config['artifacts']['artifacts_dir']
    feature_selection_dir = config['artifacts']['feature_selection_dir']
    feature_selection_df = config['artifacts']['after_selection_data']

    main_data_file_path = os.path.join(artifacts_dir,feature_selection_dir,feature_selection_df)
    df = pd.read_csv(main_data_file_path)
    #read params file
    split_ratio = params['base']['test_size']
    random_state = params['base']['random_state']
    train,test = train_test_split(df,test_size=split_ratio,random_state=random_state)
    
    split_data_dir = config['artifacts']['split_data_dir']
    train_data_filename = config['artifacts']['train']
    test_data_filename = config['artifacts']['test']

    create_dir([os.path.join(artifacts_dir,split_data_dir)])

    train_data_path = os.path.join(artifacts_dir,split_data_dir,train_data_filename)
    test_data_path = os.path.join(artifacts_dir,split_data_dir,test_data_filename)

    for data,data_path in (train,train_data_path),(test,test_data_path):
        save_local_df(data,data_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_arg = args.parse_args()
    split_data(config_path=parsed_arg.config, params_path=parsed_arg.params)