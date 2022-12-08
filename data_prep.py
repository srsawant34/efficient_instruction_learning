import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class DatasetReader():
    def __init__(self, taskname, num_examples, random_seed=42):
        self.random_seed = random_seed
        self.homepath = '/data/data/ssawan13/natural_instructions/natural-instructions/tasks/'
        self.taskname = taskname
        self.tasknametosave = self.taskname.split('_')[0]
        self.num_examples = num_examples
        self.data = None
        self.notEnoughSamplesTasks = []
        
    def read_json(self,taskname):
        datapath = f'{self.homepath}{self.taskname}.json'
        with open(datapath) as file:
            data = json.load(file)
        return data

    def getExamples(self,data,which,num_examples=2):
        keys = list(data[0].keys())
        s = ''
        if num_examples:
            for i in range(num_examples):
                srow = f"{which} example {i+1}-\n{keys[0]}: {data[i][keys[0]]}\n{keys[1]}: {data[i][keys[1]]}"
                s += srow + '\n'
            return s
        else:
            return ""
    
    def getRows(self,data,num_examples, onlyIPOP=False):
        definition = data['Definition'][0]
        positives = data['Positive Examples']
        negatives = data['Negative Examples']
        instances = data['Instances']
        pos = self.getExamples(positives,'Positive',num_examples)

        numRows = len(instances)
        processed = []

        if onlyIPOP:
            for i in (range(numRows)):
                row = "input: "+instances[i]['input'] + "\noutput: "
                output = np.random.choice(instances[i]['output'])
                processed.append([row,output])
        else:
            for i in (range(numRows)):
                row = "Definition: "+ definition + "\n" + pos+"Now complete the following example-\ninput: "+instances[i]['input'] + "\noutput: "
                output = np.random.choice(instances[i]['output'])
                processed.append([row,output])
        return processed
    
    def getData(self, onlyIPOP = False):
        jsondata = self.read_json(self.taskname)
        self.data = self.getRows(jsondata, self.num_examples, onlyIPOP)
        return self.data
    
    def getCSV(self,):
        if not self.data:
            self.data= self.getData()
        df = pd.DataFrame(self.data)
        df.columns = ['input', 'output']
        path_out_csv = f'/data/data/ssawan13/natural_instructions/natural-instructions/Experiments/tryouts/Rework/data/{self.tasknametosave}/'
        if os.path.isdir(path_out_csv):
            print("Path already exits, no need to create")
            return
        os.mkdir(path_out_csv)

        df.to_csv(f'{path_out_csv}all.csv',index=False,header=True)
        return
        
    def splitData(self,data):
        n = len(data)
        task_test = []
        task_train = []
        task_val = []
        if n>=120:
            task_test = data[:100]
            task_train = data[110:]
            task_val = data[100:110]
        else: 
            task_test = data
            task_train = data
            task_val = data
            self.notEnoughSamplesTasks = [self.taskname, n]
        
        return task_train,task_val,task_test
        
    def splitandSave(self,path_out_csv, data, onepercent = False, thousand=False, twohundred=False, hundred=False, ten = False):
        path_out_csv = os.path.join(os.getcwd(),'data') if path_out_csv=="." else os.path.join(path_out_csv,'data')
        if not os.path.exists(path_out_csv):
            os.mkdir(path_out_csv)
        path_out_csv = os.path.join(path_out_csv, "tasks")
        if not os.path.exists(path_out_csv):
            os.mkdir(path_out_csv)
        

        dfPath = os.path.join(path_out_csv, self.tasknametosave)
        if not os.path.exists(dfPath):
            os.mkdir(dfPath)
        
        train,val,test = self.splitData(data)
        
        df = pd.DataFrame(train, columns=['input','output'])
        df.to_csv(os.path.join(dfPath,'train.csv'),index=False,header=True,quotechar='"')
        shape = df.shape

        if onepercent:
            one = max(1,int(shape[0]*0.01))
            onedf = df
            onedf = onedf[:one]
            onedf.to_csv(os.path.join(dfPath,'onepercent.csv'),index=False,header=True,quotechar='"')
        
        if thousand:
            thdf = df
            if shape[0]>1000:
                thdf = thdf[:1000]
            thdf.to_csv(os.path.join(dfPath,'thousand.csv'),index=False,header=True,quotechar='"')
        
        if twohundred:
            twohundf = df
            if shape[0]>200:
                twohundf = twohundf[:200]
            twohundf.to_csv(os.path.join(dfPath,'twohundred.csv'),index=False,header=True,quotechar='"')
        
        if hundred:
            hundf = df
            if shape[0]>100:
                hundf = hundf[:100]
            hundf.to_csv(os.path.join(dfPath,'hundred.csv'),index=False,header=True,quotechar='"')
        
        if ten:
            tendf = df
            tendf = tendf[:10]
            tendf.to_csv(os.path.join(dfPath,'ten.csv'),index=False,header=True,quotechar='"')
            
        df = pd.DataFrame(val, columns=['input','output'])
        df.to_csv(os.path.join(dfPath,'val.csv'),index=False,header=True,quotechar='"')
        df = pd.DataFrame(test, columns=['input','output'])
        df.to_csv(os.path.join(dfPath,'test.csv'),index=False,header=True,quotechar='"')

        return self.notEnoughSamplesTasks

def read_test_split(pathToTxt):
    test_task_list = []
    with open(pathToTxt,'r') as f:
        for line in f:
            test_task_list.append(line.strip())
    return test_task_list

def mergeDFs(which):
    types = set(["ten", "onepercent","hundred","twohundred","thousand"])
    if which not in types:
        print("Incorrect type for sample.")
        return 

    mergePath = os.path.join(os.getcwd(), "data", "merged")
    if not os.path.exists(mergePath):
        os.mkdir(mergePath)

    # Train
    print(f"Processing {which} sample data.....")
    taskPath = os.path.join(os.getcwd(), "data","tasks")
    tasks = os.listdir(taskPath)
    firstpath = os.path.join(taskPath,tasks[0])
    df = pd.read_csv(os.path.join(firstpath,f"{which}.csv"), dtype=object)
    df_merge = df
    for t in tqdm(tasks[1:]):
        path = os.path.join(taskPath, t, f"{which}.csv")
        df = pd.read_csv(path, dtype=object)
        df_merge = pd.concat([df_merge,df], ignore_index = True)
    df_merge.to_csv(os.path.join(mergePath, f"{which}.csv"), index=False,header=True,quotechar='"')
    
    # Val
    print("Processing val data.....")
    df = pd.read_csv(os.path.join(firstpath,"val.csv"), dtype=object)
    df_merge = df
    for t in tqdm(tasks[1:]):
        path = os.path.join(taskPath, t, "val.csv")
        df = pd.read_csv(path, dtype=object)
        df_merge = pd.concat([df_merge,df], ignore_index = True)
    df_merge.to_csv(os.path.join(mergePath, "val.csv"), index=False,header=True,quotechar='"')
    
    # Test
    print("Processing test data.....")
    df = pd.read_csv(os.path.join(firstpath,"test.csv"), dtype=object)
    df_merge = df
    for t in tqdm(tasks[1:]):
        path = os.path.join(taskPath, t, "test.csv")
        df = pd.read_csv(path, dtype=object)
        df_merge = pd.concat([df_merge,df], ignore_index = True)
    df_merge.to_csv(os.path.join(mergePath, "test.csv"), index=False,header=True,quotechar='"')
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=2, help = "Num of positive examples to include with Definition in instruction.")
    parser.add_argument('--only_IpOp', type=bool, default=False, help = "Boolean value that indicates to include instructions or not. Default is set to False.")
    parser.add_argument('--ten', type=bool, default=False, help = "Boolean value that indicates to create data with sample size 10. Default is set to False.")
    parser.add_argument('--onepercent', type=bool, default=False, help = "Boolean value that indicates to create data with sample size one percent of data. Default is set to False.")
    parser.add_argument('--hundred', type=bool, default=False, help = "Boolean value that indicates to create data with sample size 100. Default is set to False.")
    parser.add_argument('--twohundred', type=bool, default=True, help = "Boolean value that indicates to create data with sample size 200. Default is set to False.")
    parser.add_argument('--thousand', type=bool, default=False, help = "Boolean value that indicates to create data with sample size 1000. Default is set to True.")
    parser.add_argument('--merge', type=bool, default=False, help = "Boolean value that indicates to merge sample size of type 'which'.")
    parser.add_argument('--which', type=str, default="twohundred", help = "Which type to merge and create sample data. Default is set to twohundred. Values can be from 'ten','onepercent','hundred','twohundred','thousand','train'.")
    args = parser.parse_args()

    dataPath = os.path.join(os.getcwd(),"data")
    if not os.path.exists(dataPath):
        pathToTxt = '/data/data/ssawan13/natural_instructions/natural-instructions/splits/default/test_tasks.txt'
        test_task_list = read_test_split(pathToTxt)
        notEnoughSamples_Tasks = []
        for task in tqdm(test_task_list):
            reader = DatasetReader(task,args.num_examples)
            data = reader.getData(args.only_IpOp)
            taskName_nsample = reader.splitandSave(".",data,onepercent=args.onepercent, thousand=args.thousand, twohundred=args.twohundred, hundred=args.hundred, ten=args.ten)
            if taskName_nsample!=[]:
                notEnoughSamples_Tasks.append([taskName_nsample[0], taskName_nsample[1]])
        
        print("\nTasks with insufficient number of rows:-")
        for t,n in notEnoughSamples_Tasks:
            print(f"Task name: {t}, having {n} samples.")
    
    if args.merge:
        mergeDFs(args.which)