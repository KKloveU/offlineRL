import matplotlib.pyplot as plt
import os
import pandas as pd
log_dir='data_plot'


target_list=["Eval/AverageReturn","Eval/StdReturn"
,"Eval/MedianReturn","Eval/MinReturn","Eval/D4RL_score","training time"]

for target in target_list:
    plt.figure(figsize=(10,6))
    for data_path in os.listdir(log_dir):
        file_path=log_dir+'/'+data_path+'/'+"progress.csv"
        with open(file_path) as file:
            data=pd.read_csv(file_path,sep=',',header=0)
            plt.plot(data[target].values,label=data_path)

    plt.xlabel('step')
    plt.ylabel(target)
    plt.legend(bbox_to_anchor=(1.1, -.1),ncol=1)    
    plt.show()