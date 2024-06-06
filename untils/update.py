#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import random
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score


def data_read_breast(path,idx): 
    #print(path[0],"    ",path[1])
    if idx == 0:
       image = Image.open(path[0])
       image.convert('RGB') 
    elif idx == 1:
       image = Image.open(path[0].replace('histology_slides_resize','histology_slides_x4'))
       image.convert('RGB') 
    elif idx == 2:
       image = Image.open(path[0].replace('histology_slides_resize','histology_slides_x2'))
       image.convert('RGB') 
    elif idx == 3:
       image = Image.open(path[0].replace('histology_slides_resize','histology_slides_LR'))
       image.convert('RGB') 
       #image = image.resize((384,384),Image.ANTIALIAS)
       #image = image.resize((48,48),Image.ANTIALIAS)
       #image.convert('RGB') 
    label = int(path[1])
    apply_transform = transforms.Compose(
      [transforms.ToTensor()])
    image = apply_transform(image)
    return np.array(image), np.array(label) 



def data_read_breast_HR(path): 
    #print(path[0],"    ",path[1])
    image = Image.open(path[0])
    image.convert('RGB') 
    label = int(path[1])
    apply_transform = transforms.Compose(
      [transforms.ToTensor()])
    image = apply_transform(image)
    return np.array(image), np.array(label) 



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, path, task, idx):
        self.dataset = path[:]
        self.task = task
        self.idx = idx
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item): 
        if self.task == "breast":
            image, label = data_read_breast(self.dataset[item],self.idx)
        if self.task == "MH-breast-HR":
            image, label = data_read_breast_HR(self.dataset[item])
        if self.task == "PMR":
            image, label = data_read_PMR(self.dataset[item])
        #print(image.dtype)
        #print(torch.tensor(image))
        return image, label



class DatasetSplit_train(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, path, task, idxs):
        self.dataset = path[:]
        self.task = task
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item): 
        if self.task == "breast":
            image, label = data_read_breast(self.dataset[self.idxs[item]],self.idx)
        if self.task == "MH-breast-HR":
            image, label = data_read_breast_HR(self.dataset[self.idxs[item]])
        if self.task == "PMR":
            image, label = data_read_PMR(self.dataset[item])
        #print(image.dtype)
        #print(torch.tensor(image))
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset_train, dataset_test, logger, idx):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test,idx)
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        #'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.idx = idx
        #print(self.args.task)

    def train_test(self, dataset_train, dataset_test, idx):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        trainloader = DataLoader(DatasetSplit(dataset_train,self.args.task, idx),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset_test,self.args.task, idx),batch_size=1, shuffle=True)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                #print(batch_idx, len(images))
                # if self.args.verbose and (batch_idx % 2 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        # global_round, iter, batch_idx * len(images),
                        # len(self.trainloader.dataset),
                        # 100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_head(self, net, w_glob_keys, global_round,lr=0.00001):
        net.train()
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        local_eps = self.args.local_ep
        epoch_loss = []
        num_updates = 0
        
        for name, param in net.named_parameters():
            if name in w_glob_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
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
                # if batch_idx % 2 == 0:
                    # print('\r','| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        # global_round, iter, batch_idx * len(images),
                        # len(self.trainloader.dataset),
                        # 100. * batch_idx / len(self.trainloader), loss.item()),end='')
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print("         ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, idx, task):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        with torch.no_grad():
            running_corrects_test = torch.zeros(1).squeeze().cuda()
            labels = []
            y_pred_list = [] 

            for batch_idx, dataset in enumerate(self.testloader):
                image,label = dataset
                inputs1 = Variable(image).to(self.device)
                target = Variable(label).to(self.device)
                #print("aaaaaaaaaaaaa",labels)
                # Inference
                out = model(inputs1)
                _ , prediction =  torch.max(out,1)
                running_corrects_test = running_corrects_test + torch.sum(prediction == target)
                labels.extend(target)
                y_pred_list.extend(prediction)
            A = torch.stack(labels).reshape(-1)
            B = torch.stack(y_pred_list).reshape(-1)
            m_F1_score = f1_score(A.cpu(), B.cpu(), average='macro')
        if task == "breast":
            if self.idx == 0:
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
        print("         ")


    def inference_average(self, model, idx):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        all_loss = []
        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.device)  #######Train
                target = batchLabel.to(self.device)
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
            return np.mean(dice_list),np.mean(all_loss)
            #return dice_list,all_loss



class LocalUpdate_iid(object):
    def __init__(self, args, dataset_train, dataset_test, logger, idx, idxs):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test,idx,idxs)
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        #'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.idx = idx
        self.idxs = idxs
        #print(self.args.task)

    def train_test(self, dataset_train, dataset_test, idx, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        trainloader = DataLoader(DatasetSplit_train(dataset_train,self.args.task, idxs),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset_test,self.args.task, idx),batch_size=1, shuffle=True)
        return trainloader, testloader

    def update_weights(self, model, global_round, idx):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().cuda()
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.idxs)/self.args.local_bs)+1, loss.item(), running_corrects/len(self.idxs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print(" ")
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, idx, task):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        with torch.no_grad():
            running_corrects_test = torch.zeros(1).squeeze().cuda()
            labels = []
            y_pred_list = [] 

            for batch_idx, dataset in enumerate(self.testloader):
                image,label = dataset
                inputs1 = Variable(image).to(self.device)
                target = Variable(label).to(self.device)
                # Inference
                out = model(inputs1)
                _ , prediction =  torch.max(out,1)
                running_corrects_test = running_corrects_test + torch.sum(prediction == target)
                labels.extend(target)
                y_pred_list.extend(prediction)
            A = torch.stack(labels).reshape(-1)
            #print(len(A))
            B = torch.stack(y_pred_list).reshape(-1)
            m_F1_score = f1_score(A.cpu(), B.cpu(), average='macro')
        if task == "breast":
            if self.idx == 0:
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2352,m_F1_score),end='')
        elif task == "MH-breast-HR":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(A.cpu()),m_F1_score),end='')
        print("         ")


    def inference_average(self, model, idx):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        all_loss = []
        with torch.no_grad():
            dice_list = []
            for i,dataset in enumerate(self.testloader):
                
                batchimages,batchLabel = dataset
                inputs = batchimages.to(self.device)  #######Train
                target = batchLabel.to(self.device)
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
            return np.mean(dice_list),np.mean(all_loss)
            #return dice_list,all_loss