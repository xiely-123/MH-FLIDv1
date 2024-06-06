#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from untils.options import args_parser
from untils.Update_RHM_FedTS_2 import LocalUpdate,LocalUpdate_personal,LocalUpdatepFedCS

import model_segV1.Unet_Plus as Unet_Plus
import model_segV1.Unet as Unet
import model_segV1.ResUnet as ResUnet
import model_segV1.FCN as FCN



from untils.utils import average_weights, exp_details
from untils.utils_iid_noiid import get_dataset_segV1
#from utils import get_dataset_v2, average_weights, exp_details, get_dataset_personal


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    exp_details(args)
    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    if args.gpu:
        #torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #device = 'cuda' if args.gpu else 'cpu'

    # BUILD MODEL
    num_classes = 2
    global_model_0 = Unet_Plus.NestedUNet(1,num_classes)
    global_model_1 = FCN.FNC_8S(1,num_classes)
    global_model_2 = ResUnet.ResUNet(1,num_classes)
    global_model_3 = Unet.UNet(1,num_classes)
    

    
    global_model_0.train()
    total_num_layers = len(global_model_0.state_dict().keys())
    print(total_num_layers)
    net_keys = [*global_model_0.state_dict().keys()]
    w_glob_keys = net_keys[total_num_layers - args.num_layers_keep:]
    #w_glob_keys = net_keys[:60]
    print(w_glob_keys)
    num_param_glob = 0
    num_param_local = 0
    for key in global_model_0.state_dict().keys():
        num_param_local += global_model_0.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += global_model_0.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    
    global_model_1.train()
    total_num_layers = len(global_model_1.state_dict().keys())
    print(total_num_layers)
    net_keys = [*global_model_1.state_dict().keys()]
    w_glob_keys = net_keys[total_num_layers - args.num_layers_keep:]
    #w_glob_keys = net_keys[:60]
    print(w_glob_keys)
    num_param_glob = 0
    num_param_local = 0
    for key in global_model_1.state_dict().keys():
        num_param_local += global_model_1.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += global_model_1.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    
    global_model_2.train()
    total_num_layers = len(global_model_2.state_dict().keys())
    print(total_num_layers)
    net_keys = [*global_model_2.state_dict().keys()]
    w_glob_keys = net_keys[total_num_layers - args.num_layers_keep:]
    #w_glob_keys = net_keys[:60]
    print(w_glob_keys)
    num_param_glob = 0
    num_param_local = 0
    for key in global_model_2.state_dict().keys():
        num_param_local += global_model_2.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += global_model_2.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    
    global_model_3.train()
    total_num_layers = len(global_model_3.state_dict().keys())
    print(total_num_layers)
    net_keys = [*global_model_3.state_dict().keys()]
    w_glob_keys = net_keys[total_num_layers - args.num_layers_keep:]
    #w_glob_keys = net_keys[:60]
    print(w_glob_keys)
    num_param_glob = 0
    num_param_local = 0
    for key in global_model_3.state_dict().keys():
        num_param_local += global_model_3.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += global_model_3.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    
    

    ######HR IID 
    net_local_list = []
    net_local_list.append(copy.deepcopy(global_model_0))
    net_local_list.append(copy.deepcopy(global_model_1))
    net_local_list.append(copy.deepcopy(global_model_2))
    net_local_list.append(copy.deepcopy(global_model_3))
# training
    train_loss, train_accuracy = [], []
    for iter in range(args.epochs):
        w_glob = {}
        loss_locals = []
        if args.task == "PMR":
           idxs_users = [0,1,2,3]#,4,5,6,7,8]
        w_keys_epoch = w_glob_keys
        m = len(idxs_users)
        print(idxs_users,args.lr,args.lr_s,m)
        print("first stage")
        for idx in idxs_users:
            train_dataset, test_dataset = get_dataset_segV1(idx,args.task)
            local = LocalUpdatepFedCS(args, train_dataset, test_dataset, idx)
            net_local = net_local_list[idx]
            #w_local, loss = local.train_backbone(net_local.to(args.device), w_glob_keys, iter, idx)
            w_local, loss = local.train_backbone(net_local.to(args.device), w_glob_keys, iter, idx)
        
        if (iter+1) % 1 == 0:
            for idx in idxs_users:
                train_dataset, test_dataset = get_dataset_segV1(idx,args.task)
                net_local_test = copy.deepcopy(net_local_list[idx])
                #train_dataset, test_dataset = get_dataset(args.task)
                local_model = LocalUpdatepFedCS(args, train_dataset, test_dataset, idx)
                local_model.inference(net_local_test,idx=idx, task=args.task)
        print("second stage")
        for idx in idxs_users:
            #train_dataset, test_dataset = get_dataset(args.task)
            train_dataset, test_dataset = get_dataset_segV1(idx,args.task)
            local = LocalUpdatepFedCS(args, train_dataset, test_dataset, idx)
            net_local = net_local_list[idx]
            #w_local, loss = local.train_backbone(net_local.to(args.device), w_glob_keys, iter, idx)
            w_local, loss = local.train_TS(net_local.to(args.device), w_glob_keys, iter, idx)
            # sum up weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_local[k]
        
        if (iter+1) % 1 == 0:
            for idx in idxs_users:
                train_dataset, test_dataset = get_dataset_segV1(idx,args.task)
                net_local_test = copy.deepcopy(net_local_list[idx])
                #train_dataset, test_dataset = get_dataset(args.task)
                local_model = LocalUpdatepFedCS(args, train_dataset, test_dataset, idx)
                local_model.inference(net_local_test,idx=idx, task=args.task)
        # get weighted average for global weights
        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], m)
        
        # copy weights to each local model
        for idx in range(args.num_users):
            ###print(args.num_users)
            net_local = net_local_list[idx]
            w_local = net_local.state_dict()
            for k in w_keys_epoch:
                #print(w_local.device,w_glob.device)
                w_local[k] = w_glob[k].to(args.device) #0.6*w_local[k].to(args.device)+0.4*
            net_local.load_state_dict(w_local)
        args.lr = args.lr*0.95
        args.lr_s = args.lr_s*0.9