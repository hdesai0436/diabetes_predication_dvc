import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,create_dir,save_local_df,fill_missing_value
import os
import numpy as np


def clean_data(config_path):
    config = read_yaml(config_path)
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    raw_local_file = config['artifacts']['raw_local_file']

    #null values folder
    null_data_dir = config['artifacts']['null_dir']
    null_data_file = config['artifacts']['null_local_file']

    # clean data folder
    clean_data_dir = config['artifacts']['clean_data_dir']
    fill_missing_data_file = config['artifacts']['fill_missing_value_df']

    null_value_dir_path = os.path.join(artifacts_dir,null_data_dir)
    create_dir([null_value_dir_path])
    null_value_file_path = os.path.join(null_value_dir_path,null_data_file)

    #get raw local file
    raw_local_file_path=os.path.join(artifacts_dir,raw_local_dir,raw_local_file)
    df = pd.read_csv(raw_local_file_path)
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    null_values_dataframe = df.isnull().sum()
    null_values_dataframe.to_csv(null_value_file_path,sep=',')

    # filling missing value with median.

    df = fill_missing_value(df)

    clean_path_dir = os.path.join(artifacts_dir,clean_data_dir)
    create_dir([clean_path_dir])
    fill_missing_value_df_path = os.path.join(clean_path_dir,fill_missing_data_file)
    df.to_csv(fill_missing_value_df_path, sep=',', index=False)



if __name__ == "__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    parsed_arg=args.parse_args()
    clean_data(parsed_arg.config)