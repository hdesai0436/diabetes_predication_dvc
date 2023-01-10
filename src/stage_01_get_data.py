from src.utils.all_utils import read_yaml,create_dir
import pandas as pd
import argparse
import os

def get_data(config_path):
    config = read_yaml(config_path)
    remote_data = config['data_source']
    df = pd.read_csv(remote_data)

    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    raw_local_file = config['artifacts']['raw_local_file']

    #create path
    raw_local_dir_path = os.path.join(artifacts_dir,raw_local_dir)
    raw_local_file_path = os.path.join(raw_local_dir_path,raw_local_file)

    #create folder
    create_dir([raw_local_dir_path])
    df.to_csv(raw_local_file_path,sep=',',index=False)



if __name__ == "__main__":
    args= argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_arg = args.parse_args()

    get_data(parsed_arg.config)