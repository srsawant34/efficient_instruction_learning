import pandas as pd
import json
import os
import sys

def findAverage(path):
    path1 = os.path.join(path,'outputs')
    tasks = os.listdir(path1)
    print("Path is: ",len(tasks))
    df_res = []
    df_res_task = []
    for task in tasks:
        p = os.path.join(path1,task,'eval','results.json')
        with open(p) as json_file:
            temp = json.load(json_file)
            tname = task.split('_')[1]
            df_res.append(temp.copy())
            temp['task'] = tname
            df_res_task.append(temp)
    if not os.path.isdir(os.path.join(path,'results')):
        os.mkdir(os.path.join(path,'results'))
    df_res = pd.DataFrame(df_res)
    df_res_task = pd.DataFrame(df_res_task)
    
    avg = {}
    for col in df_res.columns:
        avg[col] = df_res[col].mean()
    avg = pd.DataFrame([avg])

    df_res.to_csv(path+'/results/eval_all_results.csv',index=False)
    df_res_task.to_csv(path+'/results/eval_withtask_results.csv',index=False)
    avg.to_csv(path+'/results/eval_avg_results.csv',index=False)

    return

if __name__=='__main__':
    testtype = sys.argv[1]
    path = os.path.join('',testtype)
    findAverage(path)