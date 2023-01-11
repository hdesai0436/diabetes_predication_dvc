import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,create_dir,save_local_df,feature_selection
import os
import numpy as np
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.feature_selection import chi2


def feature_selection_df(config_path):
    config = read_yaml(config_path)
    artifacts_dir = config['artifacts']['artifacts_dir']
    clean_data_dir = config['artifacts']['clean_data_dir']
    fill_missing_value_file = config['artifacts']['fill_missing_value_df']

    # get filling missing data
    fill_missing_df_path = os.path.join(artifacts_dir,clean_data_dir,fill_missing_value_file)
    df = pd.read_csv(fill_missing_df_path)
    feature_df, feature_important_score = feature_selection(df) #select best 6 feature from the dataset
    #feature selection file
    feature_selection_dir = config['artifacts']['feature_selection_dir']
    feature_selection_score_path = config['artifacts']['feature_selection_score']
    feature_selection_df_path = config['artifacts']['after_selection_data']

    feature_selection_dir_path = os.path.join(artifacts_dir,feature_selection_dir)
    create_dir([feature_selection_dir_path])
    feature_important_file_path = os.path.join(feature_selection_dir_path,feature_selection_score_path)
    feature_selection_df_file = os.path.join(feature_selection_dir_path,feature_selection_df_path)
    save_local_df(feature_important_score,feature_important_file_path)
    save_local_df(feature_df,feature_selection_df_file)

    



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    parsed_arg = args.parse_args()
    feature_selection_df(parsed_arg.config)