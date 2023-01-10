import yaml
import os
import numpy as np

def read_yaml(path_to_yaml:str):
    with open(path_to_yaml) as yaml_file:
        content=yaml.safe_load(yaml_file)
    return content

def create_dir(dirs:list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)
        print(f"dir is created at the {dir_path}")


def save_local_df(data,data_path):
    data.to_csv(data_path,index=False)


def fill_missing_value(dataframe):
    df = dataframe
    a = df.columns[df.isna().any()].tolist()
    for i in a:
        temp = df[df[i].notnull()]
        temp = temp[[i, 'Outcome']].groupby(['Outcome'])[[i]].median().reset_index()
        df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = temp[i].iloc[0]
        df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = temp[i].iloc[1]
    return df
