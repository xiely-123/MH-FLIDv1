#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import numpy as np





def get_dataset(task):
    if task == "breast":
        SEED =1
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        train_dataset = np.loadtxt("/home/xly/pFred_medical_identify/data_list/HR_groudtruth_train_classification_8.txt", dtype=str, delimiter='  ')#[:100]
        test_dataset = np.loadtxt("/home/xly/pFred_medical_identify/data_list/HR_groudtruth_test_classification_8.txt", dtype=str, delimiter='  ')#[:100]
        return train_dataset, test_dataset
    if task == "EndoPolyp":
        SEED =1
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
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
    if task == "PMR":
        SEED =1
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        
        
        
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
        if idx == 4:
           train_dataset = train_dataset5
           test_dataset = test_dataset5
        if idx == 5:
           train_dataset = train_dataset6
           test_dataset = test_dataset6
        if idx == 6:
           train_dataset = train_dataset7
           test_dataset = test_dataset7
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
