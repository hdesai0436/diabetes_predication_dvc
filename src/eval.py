import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,load_model,create_dir,save_local_df
import os
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,precision_recall_curve
import json
import math
EVAL_PATH = "eval"

def evaluate(config_path):
    config = read_yaml(config_path)
    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    test_data = config['artifacts']['test']

    model_dir = config['artifacts']['model']['model_dir']
    model_file = config['artifacts']['model']['model_file']

    test_path = os.path.join(artifacts_dir,split_data_dir,test_data)
    model_file_path = os.path.join(artifacts_dir,model_dir,model_file)
    df = pd.read_csv(test_path)
    x = df.drop('Outcome',axis=1)
    y = df['Outcome']


    pipe = load_model(model_file_path)
    pred = pipe.predict(x)
    roc_auc = roc_auc_score(pred, y)
    
    with open("metrics.json", "w") as rf:
        json.dump({
            'roc_auc_score':roc_auc
        },rf)




    precision, recall, prc_thresholds = precision_recall_curve(y, pred)
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    
   
    with open("roc.json", "w") as fd:
        json.dump(
            {
              "prc": [
                   {"precision": float(p), "recall": float(r), "threshold": float(t)}
                    for p, r, t in prc_points
                   
                ]

            },
            fd,
            indent=4,
            
        )









if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
   
    parsed_arg = args.parse_args()
    evaluate(config_path=parsed_arg.config)

