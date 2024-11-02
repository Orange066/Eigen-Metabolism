import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# from pyimzML.pyimzml.ImzMLParser import ImzMLParser
# import pandas as pd
# from tqdm import tqdm
# from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import os
import torch

# from torch.utils.tensorboard import SummaryWriter


normal_files = ['YQF-N-T/YQF-pos-normal_mucosa-1.txt']
tumor_files = ['YQF-N-T/YQF-pos-tumor-1.txt','YQF-N-T/YQF-pos-tumor-2.txt','YQF-N-T/YQF-pos-tumor-3.txt']
classes = [[1,0],[0,1]]

def splitDataAll():
    df = pd.read_pickle('YQF-pos.pkl')
    normal_index = []
    tumor_index = []
    for file in normal_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split( )
                if(line[3] == "1"):
                    normal_index.append([int(line[1]),int(line[2])])
        f.close()
    for file in tumor_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split( )
                if(line[3] == "1"):
                    tumor_index.append([int(line[1]),int(line[2])])
        f.close()

    all_array = []
    count = 0
    for index in normal_index:
        try:
            intensity = df["YQF-ALL-pos-{}-{}".format(index[0],index[1])]
        except Exception as e:
            continue
        # mzArray, intensity = p.getspectrum(index[0] * index[1])
        # sums = 0
        # for i in intensity:
        #     sums += i
        # if sums == 0:
        #     count += 1
        intensity = np.concatenate((intensity,[1]),axis=0)
        all_array.append(intensity)

    for index in tumor_index:
        try:
            intensity = df["YQF-ALL-pos-{}-{}".format(index[0],index[1])]
        except Exception as e:
            continue
        # sums = 0
        # for i in intensity:
        #     sums += i
        # if sums == 0:
        #     count += 1
        #     print(index[0],index[1])
        intensity = np.concatenate((intensity,[0]),axis=0)
        all_array.append(intensity)
    print(len(all_array))
    np.save('data/all_data.npy',all_array)

def Load2Dataset(allData):
    allData = np.load(allData)
    n_data = len(allData)
    train_size = int(n_data * 0.6)
    valid_size = int(n_data * 0.3)
    test_size = n_data - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(allData, [train_size, valid_size, test_size])
    x_train = np.array(train_dataset)[:,:-1]
    x_valid = np.array(valid_dataset)[:,:-1]
    x_test = np.array(test_dataset)[:,:-1]
    y_train = np.array(train_dataset)[:,-1:]
    y_valid = np.array(valid_dataset)[:,-1:]
    y_test = np.array(test_dataset)[:,-1:]
    # print(y_test.shape)
    # print(y_train.shape)
    np.save('data/x_train.npy',x_train)
    np.save('data/x_valid.npy',x_valid)
    np.save('data/x_test.npy',x_test)
    np.save('data/y_train.npy',y_train)
    np.save('data/y_valid.npy',y_valid)
    np.save('data/y_test.npy',y_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


    

class MyDataSet(Dataset):
    """ 
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)
    def __getitem__(self,index):
        return self.exp[index],self.label[index]
    def __len__(self):
        return self.len
    
def splitData():
    all_data = []
    all_label = []
    filePath = 'data/'
    files = os.listdir(filePath)
    for file in files:
        file_name = os.path.join(filePath,file)
        file_npy = np.load(file_name)
        for f in file_npy:
            all_data.append(f)
    n_data = len(all_data)
    train_size = int(n_data * 0.5)
    valid_size = int(n_data * 0.2)
    test_size = n_data - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, valid_size, test_size])
    count = 1
    for i in train_dataset:
        if all(i) == 0:
            print(i)
            break
    print(count)
    # print(len(all_data))
    


# splitDataAll()
# Load2Dataset('data/all_data.npy')
