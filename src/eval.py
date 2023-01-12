import argparse
import pandas as pd
from src.utils.all_utils import read_yaml,load_model,create_dir,save_local_df
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
    df = pd.read_csv(test_path)
    x = df.drop('Outcome',axis=1)
    y = df['Outcome']


    pipe = load_model(model_file_path)
    pred = pipe.predict(x)

    print(type(pred))
    print(type(y))

   
    con_dir = os.path.join(os.getcwd(),EVAL_PATH)
    os.makedirs(con_dir, exist_ok=True)
    actual_data_file = 'data.csv'
    actual_data_file_path = os.path.join(con_dir,actual_data_file)
    y.to_csv(actual_data_file_path,sep=",",index=False)

    predicated_data_file = 'pred.csv'
    pred_file_path = os.path.join(con_dir,predicated_data_file)
    pred_df = pd.DataFrame(pred,columns=['predication'])
    pred_df.to_csv(pred_file_path,index=False)
    



    roc_auc = roc_auc_score(pred, y)
    
    with open("metrics.json", "w") as rf:
        json.dump({
            'roc_auc_score':roc_auc
        },rf)




    r_fpr,r_tpr, th = roc_curve(y, pred)
    
    prc_points = list(zip(r_fpr, r_tpr, th))
    
   
    with open("roc.json", "w") as fd:
        json.dump(
            {
              "prc": [
                   {"r_fpr": float(p), "r_tpr": float(r), "rh": float(t)}
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

