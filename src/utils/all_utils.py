import yaml
import os
import numpy as np
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.feature_selection import chi2
import pandas as pd
import joblib

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


def TurkyOutliers(df_out,nameOfFeature,drop=False):
    
    valueOfFeature = df_out[nameOfFeature]
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(valueOfFeature, 25.)

    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(valueOfFeature, 75.)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*1.5
    # print "Outlier step:", step
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values
    # df[~((df[nameOfFeature] >= Q1 - step) & (df[nameOfFeature] <= Q3 + step))]


    # Remove the outliers, if any were specified
    print ("Number of outliers (inc duplicates): {} and outliers: {}".format(len(outliers), feature_outliers))
    if drop:
        good_data = df_out.drop(df_out.index[outliers]).reset_index(drop = True)
        print ("New dataset with removed outliers has {} samples with {} features each.".format(*good_data.shape))
        return good_data
    else: 
        print ("Nothing happens, df.shape = ",df_out.shape)
        return df_out


def feature_selection(data):
    df = data
    X = df.drop('Outcome',axis=1)
    y = df['Outcome']
    ordered_rank_features=SelectKBest(score_func=chi2,k=6)
    ordered_feature=ordered_rank_features.fit(X,y)
    dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
    dfcolumns=pd.DataFrame(X.columns)
    features_rank=pd.concat([dfcolumns,dfscores],axis=1)
    features_rank.columns=['Features','Score']
    best_feature_6 = features_rank.nlargest(6,'Score')['Features']
    col_name = list(best_feature_6) # convert pandas series to list 
    col_name.append('Outcome') # append target columns to list
    feature_df = df.loc[:, df.columns.isin(list(col_name))] # choose only select columns from pandas dataframe
    return feature_df, features_rank

def save_model( model, model_filename):
    joblib.dump(model,model_filename)

def load_model(model_file_path):
    model = joblib.load(model_file_path)
    return model