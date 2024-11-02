import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import pandas as pd
import pickle
import os
import torch
xlsx_path = "/home/user8/work/TumorAnalysis/MoreData/original"
def xlsx2pkl(xlsx_path):
    all_xlsx = os.listdir(xlsx_path)
    for i in all_xlsx:
        all_file = os.path.join(xlsx_path,i)
        print(all_file)
        name = i.split("xlsx")[0]
        df = pd.read_excel(all_file)
        # 将数据保存为.pkl文件
        pkl_file_path = '/home/user8/work/TumorAnalysis/MoreData/allPkl/{}pkl'.format(name)
        df.to_pickle(pkl_file_path)

normal_files = ['MoreData/1/YQF-pos-normal_mucosa-1.txt']
tumor_files = ['MoreData/1/YQF-pos-tumor-1.txt','MoreData/1/YQF-pos-tumor-2.txt','MoreData/1/YQF-pos-tumor-3.txt']
subnormal_files = ['MoreData/1/YQF-pos-normal_submucosa-1.txt']
tumor_stroma_files = ['MoreData/1/YQF-pos-tumor_stroma-1.txt','MoreData/1/YQF-pos-tumor_stroma-2.txt']
YQF_file_pkl = 'MoreData/allPkl/YQF-ALL-pos-pixel_level-pos.pkl'

def normalize(ar):
    min_v = np.min(ar)
    max_v = np.max(ar)
    for i in ar:
        i = (i - min_v) / (max_v - min_v)

    return ar

def readFromTxt(txt_file):
    result = []
    for file in txt_file:
        flag = False
        temp = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split( )
                if line[3] == "1":
                    flag = True
                    temp.append([int(line[1]),int(line[2])])
                if line[3] == '-1':
                    flag = False
                if(len(temp) > 0 and flag == False):
                    result.append(temp)
                    temp = []
        f.close()
    return result

def pkl2npy():
    df = pd.read_pickle(YQF_file_pkl)
    
   
    normal = readFromTxt(normal_files)
    subnormal = readFromTxt(subnormal_files)
    tumor_stro = readFromTxt(tumor_stroma_files)
    tumor = readFromTxt(tumor_files)
    
    all_array = []
    for patch in normal:
        temp = []
        for index in patch:
            try:
                intensity = df["YQF-ALL-pos-{}-{}".format(index[0],index[1])]
            except Exception as e:
                print("1")
                continue
            intensity = np.concatenate((intensity,[0]),axis=0)
            temp.append(intensity)
        all_array.append(temp)

    for patch in subnormal:
        temp = []
        for index in patch:
            try:
                intensity = df["YQF-ALL-pos-{}-{}".format(index[0],index[1])]
            except Exception as e:
                print("2")
                continue
            intensity = np.concatenate((intensity,[1]),axis=0)
            temp.append(intensity)
        all_array.append(temp)

    for patch in tumor_stro:
        temp = []
        for index in patch:
            try:
                intensity = df["YQF-ALL-pos-{}-{}".format(index[0],index[1])]
            except Exception as e:
                print("3")
                continue
            intensity = np.concatenate((intensity,[2]),axis=0)
            temp.append(intensity)
        all_array.append(temp)

    for patch in tumor:
        temp = []
        for index in patch:
            try:
                intensity = df["YQF-ALL-pos-{}-{}".format(index[0],index[1])]
            except Exception as e:
                print("4")
                continue
            intensity = np.concatenate((intensity,[3]),axis=0)
            temp.append(intensity)
        all_array.append(temp)

    print(len(all_array))
    arr = np.array(all_array,dtype=object)
    np.save('MoreData/1/all_data.npy',arr)
# xlsx2pkl(xlsx_path)

# pkl2npy()
# file = np.load('MoreData/1/all_data.npy',allow_pickle=True)
# print(len(file[0]))
def dealDataset(dataset, partial = None, norm = False):
    x = []
    y = []
    print(norm)
    if partial != None:
        print("partial")
        lis = [i[0] for i in partial.most_common()]
        for patch in dataset:
            for index in patch:
                temp = []
                for i in lis:
                    temp.append(index[i])
                if norm:
                    temp = normalize(temp)
                x.append(temp)
                y.append(index[-1:])
    else:
        print("all")
        for patch in dataset:
            for index in patch:
                temp = index[:-1]
                if norm:
                    temp = normalize(temp)
                x.append(temp)
                y.append(index[-1:])
   
    return np.array(x), np.array(y)

def load2Dataset(allData, norm = False,partial=None):
    allData = np.load(allData,allow_pickle=True)
    print(norm)
    n_data = len(allData)
    print(n_data)
    train_size = int(n_data * 0.6)
    valid_size = int(n_data * 0.3)
    test_size = n_data - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(allData, [train_size, valid_size, test_size])
    # print(len(train_dataset))
    print("yes")
    x_train, y_train = dealDataset(train_dataset, partial=partial,norm=norm)
    x_valid, y_valid = dealDataset(valid_dataset, partial=partial,norm=norm)
    x_test, y_test = dealDataset(test_dataset, partial=partial,norm=norm)
    np.save('MoreData/1/x_train2.npy',x_train)
    np.save('MoreData/1/x_valid2.npy',x_valid)
    np.save('MoreData/1/x_test2.npy',x_test)
    np.save('MoreData/1/y_train2.npy',y_train)
    np.save('MoreData/1/y_valid2.npy',y_valid)
    np.save('MoreData/1/y_test2.npy',y_test)


def dealwithPartialList(path_to_npy,partial_list):
    partial_list = np.load(partial_list)
    x_train = np.load(path_to_npy + '/x_train.npy')
    x_valid = np.load(path_to_npy + '/x_valid.npy')
    x_test = np.load(path_to_npy + '/x_test.npy')
    print(len(x_train))
    print(len(x_valid))
    print(len(x_test))
    x_train_new = []
    x_valid_new = []
    x_test_new = []
    for i in x_train:
        temp = []
        for j in partial_list:
            temp.append(i[j])
        x_train_new.append(temp)

    for i in x_valid:
        temp = []
        for j in partial_list:
            temp.append(i[j])
        x_valid_new.append(temp)
    
    for i in x_test:
        temp = []
        for j in partial_list:
            temp.append(i[j])
        x_test_new.append(temp)
    print(len(x_train_new[0]))
    print(len(x_valid_new[0]))
    print(len(x_test_new[0]))
    np.save(path_to_npy + '/x_train_contrastive_p.npy', x_train_new)
    np.save(path_to_npy + '/x_valid_contrastive_p.npy', x_valid_new)
    np.save(path_to_npy + '/x_test_contrastive_p.npy', x_test_new) 
    
def shuffledataset(path_to_npy):
    x_train = np.load(path_to_npy + '/x_train.npy')
    x_valid = np.load(path_to_npy + '/x_valid.npy')
    x_test = np.load(path_to_npy + '/x_test.npy')
    print(len(x_train))
    print(len(x_valid))
    print(len(x_test))
    rng_state = np.random.get_state()
    for x in x_train:
        np.random.set_state(rng_state)
        np.random.shuffle(x)
    for x in x_valid:
        np.random.set_state(rng_state)
        np.random.shuffle(x) 
    for x in x_test:
        np.random.set_state(rng_state)
        np.random.shuffle(x)
    np.save(path_to_npy + '/x_train_shuffle.npy', x_train)
    np.save(path_to_npy + '/x_valid_shuffle.npy', x_valid)
    np.save(path_to_npy + '/x_test_shuffle.npy', x_test) 
# pkl2npy()
# load2Dataset('MoreData/1/all_data.npy',norm=False)
# load2DatasetwithPartial('MoreData/1/all_data.npy','MoreData/1/xianZhuWu.pkl',norm=True)
# dealwithPartial('MoreData/1', partial='MoreData/1/buxianZhuWu.pkl', norm=False)
dealwithPartialList('MoreData/2',"moreproject/2_simple_contrastive/mask_True.npy")
# shuffledataset('MoreData/1')
