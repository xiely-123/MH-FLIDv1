#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from untils.sampling import dataset_iid, dataset_noniid, dataset_noniid_personal, dataset_noniid_personal_OCT,dataset_noniid_personal_timeseries
import numpy as np


def get_dataset_v2(args):
    train_data_dir = "/home/xly/pFred_medical_identify/data_list/HR_groudtruth_train_classification_8.txt"
    test_data_dir = "/home/xly/pFred_medical_identify/data_list/HR_groudtruth_test_classification_8.txt"
    train_dataset = np.loadtxt(train_data_dir, dtype=str, delimiter='  ')#[57:]
    test_dataset = np.loadtxt(test_data_dir, dtype=str, delimiter='  ')
    data = np.concatenate([train_dataset,test_dataset],0)
    train_dataset = data[:6000]
    test_dataset = data[6000:]
    
    print(len(train_dataset),len(test_dataset))
    if args.iid:
        user_groups = dataset_iid(train_dataset, args.num_users)
    else: 
        
        user_groups = dataset_noniid(train_dataset, args.num_users)
    # for i in range(len(user_groups)):
        # print(len(user_groups[i]))
        
        #print(user_groups[0])
    #print(user_groups[0])
    return train_dataset, test_dataset, user_groups
    
    


def get_dataset_OCT(args):
    train_data_dir = "/home/xly/pFred_medical_identify/data_list/OCT._classification_4.txt"
    dataset = np.loadtxt(train_data_dir, dtype=str, delimiter='  ')#[57:]
    train_dataset = dataset[:40000]
    test_dataset = dataset[95000:]
    
    #print(train_dataset[0])
    if args.iid:
        user_groups = dataset_iid(train_dataset, args.num_users)
    else: 
        
        user_groups = dataset_noniid(train_dataset, args.num_users)
    # for i in range(len(user_groups)):
        # print(len(user_groups[i]))
        
        #print(user_groups[0])
    #print(user_groups[0])
    return train_dataset, test_dataset, user_groups



def get_dataset_personal(args):
    train_data_dir = "/home/xly/pFred_medical_identify/data_list/HR_groudtruth_train_classification_8.txt"
    test_data_dir = "/home/xly/pFred_medical_identify/data_list/HR_groudtruth_test_classification_8.txt"
    train_dataset = np.loadtxt(train_data_dir, dtype=str, delimiter='  ')#[57:]
    test_dataset = np.loadtxt(test_data_dir, dtype=str, delimiter='  ')
    data = np.concatenate([train_dataset,test_dataset,train_dataset],0)
    train_dataset = data[:8000]
    #test_dataset = np.concatenate([train_dataset,test_dataset],0)[:1]
    #print(len(train_dataset))
    user_groups = dataset_noniid_personal(train_dataset, args.num_users)
    # for i in range(len(user_groups)):
        # print(len(user_groups[i]))
        
        #print(user_groups[0])
    #print(user_groups[0])
    return train_dataset, user_groups



def get_dataset_OCT_noiid(args):
    train_data_dir = "/home/xly/pFred_medical_identify/data_list/OCT._classification_4.txt"
    dataset = np.loadtxt(train_data_dir, dtype=str, delimiter='  ')#[57:]
    train_dataset = dataset[:40000]
    user_groups = dataset_noniid_personal_OCT(train_dataset, args.num_users)
    # for i in range(len(user_groups)):
        # print(len(user_groups[i]))
        
        #print(user_groups[0])
    #print(user_groups[0])
    return train_dataset, user_groups





def get_dataset_seg(idx,task):
    SEED =1
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    path = "/home/xly/pFred_medical_segmentation/data_process/data_list/CVC-ColonDB.txt"
    path1 = "/home/xly/pFred_medical_segmentation/data_process/data_list/ETIS-LaribPolypDB.txt"
    path2 = "/home/xly/pFred_medical_segmentation/data_process/data_list/CVC-ClinicDB.txt"
    path3 = "/home/xly/pFred_medical_segmentation/data_process/data_list/Kvasir-SEG.txt"


    train_dataset1 = np.loadtxt(path, dtype=str, delimiter='  ')
    train_dataset2 = np.loadtxt(path1, dtype=str, delimiter='  ')
    train_dataset3 = np.loadtxt(path2, dtype=str, delimiter='  ')
    train_dataset4 = np.loadtxt(path3, dtype=str, delimiter='  ')

    permutation = np.random.permutation(train_dataset1.shape[0])####数据随机
    train_dataset1 = train_dataset1[permutation]
    test_dataset1 = train_dataset1[int(0.5*len(train_dataset1)):]
    train_dataset1 = train_dataset1[:int(0.5*len(train_dataset1))]

    permutation = np.random.permutation(train_dataset2.shape[0])####数据随机
    train_dataset2 = train_dataset2[permutation]
    test_dataset2 = train_dataset2[int(0.5*len(train_dataset2)):]
    train_dataset2 = train_dataset2[:int(0.5*len(train_dataset2))]

    permutation = np.random.permutation(train_dataset4.shape[0])####数据随机
    train_dataset4 = train_dataset4[permutation]
    test_dataset4 = train_dataset4[int(0.5*len(train_dataset4)):]
    train_dataset4 = train_dataset4[:int(0.5*len(train_dataset4))]

    permutation = np.random.permutation(train_dataset3.shape[0])####数据随机
    train_dataset3 = train_dataset3[permutation]
    test_dataset3 = train_dataset3[int(0.5*len(train_dataset3)):]
    train_dataset3 = train_dataset3[:int(0.5*len(train_dataset3))]
    #Train_list = train_dataset7#np.concatenate([train_dataset1,train_dataset2,train_dataset3,train_dataset4,train_dataset5,train_dataset6,train_dataset7],0)
    #Test_list = np.concatenate([test_dataset1,test_dataset2,test_dataset3,test_dataset4,test_dataset5,test_dataset6,test_dataset7],0)
    if idx == 0:
       train_dataset = train_dataset1
       test_dataset = test_dataset1
    if idx == 1:
       train_dataset = train_dataset2
       test_dataset = test_dataset2
    if idx == 2:
       train_dataset = train_dataset3
       test_dataset = test_dataset3
    if idx == 3:
       train_dataset = train_dataset4
       test_dataset = test_dataset4
    return train_dataset, test_dataset
    

def get_dataset_Timeseries_noiid(args):
    path = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/train.pt"
    path1 = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/val.pt"
    dataset = torch.load(path)
    dataset2 = torch.load(path1)
    X_train = dataset["samples"]#3000
    y_train = dataset["labels"]
    X_train_2 = dataset2["samples"]#3000
    y_train_2 = dataset2["labels"]
    X_train = torch.cat((X_train,X_train_2),dim=0)[:33396]
    y_train = torch.cat((y_train,y_train_2),dim=0)[:33396]
    user_groups = dataset_noniid_personal_timeseries(X_train, y_train, args.num_users)
    #print(X_train[0],y_train[0])
    # dataset = 
    # train_dataset = dataset[:40000]
    # user_groups = dataset_noniid_personal_OCT(train_dataset, args.num_users)
    # # for i in range(len(user_groups)):
        # # print(len(user_groups[i]))
        
        # #print(user_groups[0])
    # #print(user_groups[0])
    return X_train, y_train, user_groups



def get_dataset_segV1(idx,task):
        SEED =1
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        path = "/home/xly/pFred_medical_segmentation/data_process/data_list/BIDMC_train.txt"##0.8
        path1 = "/home/xly/pFred_medical_segmentation/data_process/data_list/BMC_train.txt"##0.8
        path2 = "/home/xly/pFred_medical_segmentation/data_process/data_list/HK_train.txt"
        path3 = "/home/xly/pFred_medical_segmentation/data_process/data_list/Promise12_train.txt"
        # path3 = "/home/xly/pFred_medical_segmentation/data_process/data_list/I2CVB_train.txt"##0.8
        # path4 = "/home/xly/pFred_medical_segmentation/data_process/data_list/RUNMC_train.txt"##0.8
        # path5 = "/home/xly/pFred_medical_segmentation/data_process/data_list/UCL_train.txt"


        train_dataset1 = np.loadtxt(path, dtype=str, delimiter='  ')
        train_dataset2 = np.loadtxt(path1, dtype=str, delimiter='  ')
        train_dataset3 = np.loadtxt(path2, dtype=str, delimiter='  ')
        train_dataset4 = np.loadtxt(path3, dtype=str, delimiter='  ')
        # train_dataset5 = np.loadtxt(path4, dtype=str, delimiter='  ')
        # train_dataset6 = np.loadtxt(path5, dtype=str, delimiter='  ')


        path_test = "/home/xly/pFred_medical_segmentation/data_process/data_list/BIDMC_test.txt"##0.8
        path_test1 = "/home/xly/pFred_medical_segmentation/data_process/data_list/BMC_test.txt"##0.8
        path_test2 = "/home/xly/pFred_medical_segmentation/data_process/data_list/HK_test.txt"
        path_test3 = "/home/xly/pFred_medical_segmentation/data_process/data_list/Promise12_test.txt"
        # path_test3 = "/home/xly/pFred_medical_segmentation/data_process/data_list/I2CVB_test.txt"##0.8
        # path_test4 = "/home/xly/pFred_medical_segmentation/data_process/data_list/RUNMC_test.txt"##0.8
        # path_test5 = "/home/xly/pFred_medical_segmentation/data_process/data_list/UCL_test.txt"

        test_dataset1 = np.loadtxt(path_test, dtype=str, delimiter='  ')
        test_dataset2 = np.loadtxt(path_test1, dtype=str, delimiter='  ')
        test_dataset3 = np.loadtxt(path_test2, dtype=str, delimiter='  ')
        test_dataset4 = np.loadtxt(path_test3, dtype=str, delimiter='  ')
        # test_dataset5 = np.loadtxt(path_test4, dtype=str, delimiter='  ')
        # test_dataset6 = np.loadtxt(path_test5, dtype=str, delimiter='  ')
        
        
        
        if idx == 0:
           train_dataset = train_dataset1
           test_dataset = test_dataset1
        if idx == 1:
           train_dataset = train_dataset2
           test_dataset = test_dataset2
        if idx == 2:
           train_dataset = train_dataset3
           test_dataset = test_dataset3
        if idx == 3:
           train_dataset = train_dataset4
           test_dataset = test_dataset4
        # if idx == 4:
           # train_dataset = train_dataset5
           # test_dataset = test_dataset5
        # if idx == 5:
           # train_dataset = train_dataset6
           # test_dataset = test_dataset6
        return train_dataset, test_dataset




def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
