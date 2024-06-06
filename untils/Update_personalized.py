# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit: Paul Pu Liang

# For MAML (PerFedAvg) implementation, code was adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# credit: Antreas Antoniou

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
import random
from torchvision import datasets, transforms
from PIL import Image
from untils.dataread import data_read,data_read_EndoPolyp,data_read_PMR
from untils.dice_score import dice_coeff,dice_loss,multiclass_dice_coeff
import torch.nn.functional as F

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, path, task):
        self.dataset = path[:]
        self.task = task
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item): 
        if self.task == "RIF":
            image, label = data_read(self.dataset[item])
        if self.task == "EndoPolyp":
            image, label = data_read_EndoPolyp(self.dataset[item])
        if self.task == "PMR":
            image, label = data_read_PMR(self.dataset[item])
        #print(image.dtype)
        #print(torch.tensor(image))
        return image, label

class Datasettest(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, path):
        self.dataset = path[:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = data_read(self.dataset[item])
        #print(image.dtype)
        #print(torch.tensor(image))
        return torch.tensor(image), torch.tensor(label)



class LocalUpdateScaffold(object):

    def __init__(self, args, dataset_train, dataset_test, idx):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test)
        self.idx=idx
        
    def train_test(self, dataset_train, dataset_test):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(dataset_train,self.args.task),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset_test,self.args.task),batch_size=1, shuffle=True)
        return trainloader, testloader

    def train(self, net, c_list={}, idx=-1, lr=0.1, c=False):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        local_eps = self.args.local_ep

        epoch_loss=[]
        num_updates = 0
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                out = net(images)
                loss = dice_loss(
                                    F.softmax(out, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss_fi = 0.5*loss + 0.5*self.loss_func(out, labels)
                w = net.state_dict()
                local_par_list = None
                dif = None
                for index,param in enumerate(net.state_dict()):
                    if not isinstance(local_par_list, torch.Tensor):
                        local_par_list = net.state_dict()[param].reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, net.state_dict()[param].reshape(-1)), 0)
                for k in c_list[idx].keys():
                    #print(len((-c_list[idx][k] +c_list[-1][k]).reshape(-1)),k)
                    if not isinstance(dif, torch.Tensor):
                        dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                    else:
                        dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                loss_algo = torch.sum(local_par_list * dif)
                loss = loss_fi + loss_algo
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer.step()
                #print(len(net.parameters()))
                num_updates += 1
                # if num_updates == self.args.local_updates:
                    # break
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), num_updates
    
    def inference(self, net, idx, task):
        """ Returns the inference accuracy and loss.
        """
        model = copy.deepcopy(net)
        model = model.to(self.args.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.args.device)  #######Train
                target = batchLabel.to(self.args.device)
                mask_true = target.long()
                mask_pred = model(inputs)             #得到前向传播的结果
                #print(out.size())
                mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #print(mask_true.size(),mask_pred.size())
                #print(dice, dice.item())
                dice_list.append(dice.item())
                #print(dice)
                if task == "RIF":
                    if self.idx == 0:
                        print('\r','BinRushed dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','Magrabia dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','IDRID_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','RIM-ONE dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','MESSIDOR dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','REFUGE_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 6:
                        print('\r','Drishti_GS1_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "EndoPolyp":
                    if self.idx == 0:
                        print('\r','CVC-ColonDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','ETIS-LaribPolypDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','CVC-ClinicDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','Kvasir-SEG dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "PMR":
                    if self.idx == 0:
                        print('\r','BIDMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','BMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','HK dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','I2CVB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','RUNMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','UCL dice:{:.4f}'.format(np.mean(dice_list)),end='')
            print("         ")


class LocalUpdateAPFL(object):

    def __init__(self, args, dataset_train, dataset_test, idx):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test)
        self.idx=idx
        
    def train_test(self, dataset_train, dataset_test):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(dataset_train,self.args.task),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset_test,self.args.task),batch_size=1, shuffle=True)
        return trainloader, testloader

    def train(self, net,ind=None,w_local=None, idx=-1, lr=0.1):
        net.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        for iter in range(local_eps):
            batch_loss = []
            if num_updates >= self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                w_loc_new = {} 
                w_glob = copy.deepcopy(net.state_dict())
                for k in net.state_dict().keys():
                    w_loc_new[k] = self.args.alpha_apfl*w_local[k].to(self.args.device) + self.args.alpha_apfl*w_glob[k]
                ###全局模型更新
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                out = net(images)
                loss = dice_loss(
                                    F.softmax(out, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.5*loss + 0.5*self.loss_func(out, labels)
                optimizer.zero_grad()
                loss.backward()
                    
                optimizer.step()
                wt = copy.deepcopy(net.state_dict())
                ####局部更新
                net.load_state_dict(w_loc_new)
                log_probs = net(images)
                out = net(images)
                loss = dice_loss(
                                    F.softmax(out, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.5*loss + 0.5*self.loss_func(out, labels)
                loss = self.args.alpha_apfl*loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #######
                w_local_bar = net.state_dict()
                for k in w_local_bar.keys():
                    w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k].to(self.args.device)

                net.load_state_dict(wt)
                optimizer.zero_grad()
                del wt
                del w_loc_new
                del w_glob
                del w_local_bar

                num_updates += 1
                if num_updates >= self.args.local_updates:
                    break

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(),w_local, sum(epoch_loss) / len(epoch_loss)


    def inference(self, net, idx):
        """ Returns the inference accuracy and loss.
        """
        model = copy.deepcopy(net)
        model = model.to(self.args.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.args.device)  #######Train
                target = batchLabel.to(self.args.device)
                mask_true = target.long()
                mask_pred = model(inputs)             #得到前向传播的结果
                #print(out.size())
                mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #print(mask_true.size(),mask_pred.size())
                #print(dice, dice.item())
                dice_list.append(dice.item())
                #print(dice)
                if task == "RIF":
                    if self.idx == 0:
                        print('\r','BinRushed dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','Magrabia dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','IDRID_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','RIM-ONE dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','MESSIDOR dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','REFUGE_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 6:
                        print('\r','Drishti_GS1_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "EndoPolyp":
                    if self.idx == 0:
                        print('\r','CVC-ColonDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','ETIS-LaribPolypDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','CVC-ClinicDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','Kvasir-SEG dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "PMR":
                    if self.idx == 0:
                        print('\r','BIDMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','BMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','HK dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','I2CVB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','RUNMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','UCL dice:{:.4f}'.format(np.mean(dice_list)),end='')
            print("         ")

class LocalUpdateDitto(object):

    def __init__(self, args, dataset_train, dataset_test, idx):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test)
        self.idx=idx
        
    def train_test(self, dataset_train, dataset_test):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(dataset_train,self.args.task),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset_test,self.args.task),batch_size=1, shuffle=True)
        return trainloader, testloader


    def train(self, net, global_round, w_ditto=None, lam=0, lr=0.1):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                out = net(images)
                loss = dice_loss(
                                    F.softmax(out, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.5*loss + 0.5*self.loss_func(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(batch_idx)
                if batch_idx % 2 == 0:
                    print('\r','| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()),end='')
                
                if w_ditto is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key].to(self.args.device))
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print("         ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)#, self.indd

    def inference(self, net, idx, task):
        """ Returns the inference accuracy and loss.
        """
        model = copy.deepcopy(net)
        model = model.to(self.args.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.args.device)  #######Train
                target = batchLabel.to(self.args.device)
                mask_true = target.long()
                mask_pred = model(inputs)             #得到前向传播的结果
                #print(out.size())
                mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #print(mask_true.size(),mask_pred.size())
                #print(dice, dice.item())
                dice_list.append(dice.item())
                #print(dice)
                if task == "RIF":
                    if self.idx == 0:
                        print('\r','BinRushed dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','Magrabia dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','IDRID_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','RIM-ONE dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','MESSIDOR dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','REFUGE_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 6:
                        print('\r','Drishti_GS1_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "EndoPolyp":
                    if self.idx == 0:
                        print('\r','CVC-ColonDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','ETIS-LaribPolypDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','CVC-ClinicDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','Kvasir-SEG dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "PMR":
                    if self.idx == 0:
                        print('\r','BIDMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','BMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','HK dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','I2CVB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','RUNMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','UCL dice:{:.4f}'.format(np.mean(dice_list)),end='')
            print("         ")


# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdatefedrep(object):
    def __init__(self, args, dataset_train, dataset_test, idx):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test)
        self.idx=idx
        
    def train_test(self, dataset_train, dataset_test):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(dataset_train,self.args.task),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset_test,self.args.task),batch_size=1, shuffle=True)
        return trainloader, testloader


    def train(self, net, w_glob_keys, global_round,lr=0.00001):
        net.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        local_eps = self.args.local_ep
        head_eps = local_eps-self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if iter < head_eps:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # then do local epochs for the representation
            elif iter == head_eps:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

       
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                out = net(images)
                loss = dice_loss(
                                    F.softmax(out, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.5*loss + 0.5*self.loss_func(out, labels)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if batch_idx % 2 == 0:
                    print('\r','| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()),end='')
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print("         ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, net, w_locals, w_glob_keys, idx, task, epoch):
        """ Returns the inference accuracy and loss.
        """
        model = copy.deepcopy(net)
        if w_locals is not None:
            w_local = model.state_dict()
            for k in w_locals[idx].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
                elif w_glob_keys is None:
                    w_local[k] = w_locals[idx][k]
            model.load_state_dict(w_local)
        model = model.to(self.args.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.args.device)  #######Train
                target = batchLabel.to(self.args.device)
                mask_true = target.long()
                mask_pred = model(inputs)             #得到前向传播的结果
                #print(out.size())
                mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #print(mask_true.size(),mask_pred.size())
                #print(dice, dice.item())
                dice_list.append(dice.item())
                #print(dice)
                if task == "RIF":
                    if self.idx == 0:
                        print('\r','BinRushed dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','Magrabia dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','IDRID_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','RIM-ONE dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','MESSIDOR dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','REFUGE_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 6:
                        print('\r','Drishti_GS1_Train dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "EndoPolyp":
                    if self.idx == 0:
                        print('\r','CVC-ColonDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                        torch.save(net, "/home/xly/pFred_medical_segmentation/save_model/FedRep/"+str(epoch + 1)+"my_model.pth")
                    if self.idx == 1:
                        print('\r','ETIS-LaribPolypDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','CVC-ClinicDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','Kvasir-SEG dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if task == "PMR":
                    if self.idx == 0:
                        print('\r','BIDMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 1:
                        print('\r','BMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 2:
                        print('\r','HK dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 3:
                        print('\r','I2CVB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 4:
                        print('\r','RUNMC dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    if self.idx == 5:
                        print('\r','UCL dice:{:.4f}'.format(np.mean(dice_list)),end='')
            print("         ")


    def inference_average(self, net, w_locals, w_glob_keys, idx):
        """ Returns the inference accuracy and loss.
        """
        model = copy.deepcopy(net)
        if w_locals is not None:
            w_local = model.state_dict()
            for k in w_locals[idx].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
                elif w_glob_keys is None:
                    w_local[k] = w_locals[idx][k]
            model.load_state_dict(w_local)
        model = model.to(self.args.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        all_loss = []
        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.args.device)  #######Train
                target = batchLabel.to(self.args.device)
                mask_true = target.long()
                mask_pred = model(inputs)             #得到前向传播的结果
                loss = dice_loss(
                                    F.softmax(mask_pred, dim=1).float(),
                                    F.one_hot(mask_true,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.5*loss + 0.5*self.loss_func(mask_pred, mask_true)
                
                
                #print(out.size())
                mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #print(mask_true.size(),mask_pred.size())
                #print(dice, dice.item())
                dice_list.append(dice.item())
                all_loss.append(loss.item())
                #print(dice)
            #return dice_list,all_loss
            return np.mean(dice_list),np.mean(all_loss)

