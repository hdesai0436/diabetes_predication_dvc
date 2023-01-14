import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,load_model,create_dir,save_local_df,save_json
import os
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,precision_recall_curve
import json
import math
EVAL_PATH = "confusion"



def evaluate(config_path):
    config = read_yaml(config_path)
    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    test_data = config['artifacts']['test']

    model_dir = config['artifacts']['model']['model_dir']
    model_file = config['artifacts']['model']['model_file']

    test_path = os.path.join(artifacts_dir,split_data_dir,test_data)
    model_file_path = os.path.join(artifacts_dir,model_dir,model_file)

    prc_json_path = config['plots']['PRC']
    roc_json_path = config['plots']['ROC']
    scores_json_path = config['metrics']['SCORES']

    df = pd.read_csv(test_path)
    x = df.drop('Outcome',axis=1)
    y = df['Outcome']


    pipe = load_model(model_file_path)
    pred = pipe.predict(x)

    roc_auc = roc_auc_score(y,pred)
    scores = {
        "roc_auc":roc_auc
    }
    save_json(scores_json_path,scores)

    precision,recall,prc_th = precision_recall_curve(y,pred)

    nth_point = math.ceil(len(prc_th)/1000)
    prc_points =list(zip(precision,recall,prc_th))[::nth_point]
    prc_data = {
        'prc':[
            {"precision": float(p), "recall": float(r), "thresh": float(t)}
            for p,r,t in prc_points
        ]
    }
    save_json(prc_json_path,prc_data)


    fpr,tpr,roc_th = roc_curve(y,pred)

    roc_data = {
        'roc': [
            {'fpr':float(fp), "tpr":float(tp),"th":float(t)}
            for fp,tp,t in zip(fpr,tpr,roc_th)
        ]
    }
    save_json(roc_json_path,roc_data)



    


    
    



    

    

    









if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
   
    parsed_arg = args.parse_args()
    evaluate(config_path=parsed_arg.config)

