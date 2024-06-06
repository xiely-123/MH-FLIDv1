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
import copy
from untils.dice_score import dice_coeff,dice_loss,multiclass_dice_coeff
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

def data_read_OCT(path): 
    #print(path[0],"    ",path[1])
    image = Image.open(path[0])
    image.convert('RGB') 
    image = image.resize((256,256),Image.ANTIALIAS)
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
        if self.task == "breast-DR":
            image, label = data_read_breast(self.dataset[item],self.idx)
        if self.task == "MH-breast-HR":
            image, label = data_read_breast_HR(self.dataset[item])
        if self.task == "Messidor":
            image, label = data_read_Messidor(self.dataset[item])
        if self.task == "OCT":
            image, label = data_read_OCT(self.dataset[item])
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
        if self.task == "Messidor":
            image, label = data_read_Messidor(self.dataset[self.idxs[item]])
        if self.task == "OCT":
            image, label = data_read_OCT(self.dataset[self.idxs[item]])
        #print(image.dtype)
        #print(torch.tensor(image))
        return image, label




def kd_ce_Loss(Logits_S, Logits_T, temperature=1):
    '''
    Calculate the cross entropy between Logits_S and Logits_T

    :param Logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param Logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_Logits_T = Logits_T / temperature
    beta_Logits_S = Logits_S / temperature
    p_T = F.softmax(beta_Logits_T, dim=-1)
    Loss = -(p_T * F.log_softmax(beta_Logits_S, dim=-1)).sum(dim=-1).mean()
    return Loss













class LocalUpdate(object):
    def __init__(self, args, dataset_train, dataset_test, logger, idx, idxs):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_test,idx,idxs)
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        #'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.MSE = nn.MSELoss().to(self.device)
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
                log_probs, log_probs_S = model(images)
                loss = self.criterion(log_probs, labels)
                loss_S = self.criterion(log_probs_S, labels)
                loss = 0.1*loss_S + 0.9*loss
                loss.backward()
                optimizer.step()
                #print(batch_idx, len(images))
                if self.args.verbose and (batch_idx % 2 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)




    def train_backbone(self, net, w_glob_keys, global_round, idx):
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
        
        for name, param in net.named_parameters(): #####冻结Student
            if name in w_glob_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for iter in range(local_eps):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                net.zero_grad()
                log_probs, log_probs_S = net(images,1)
                loss = self.criterion(log_probs, labels)
                loss_S = self.criterion(log_probs_S, labels)
                #loss_2 = self.criterion(log_probs_2, labels)
                loss = 0.1*loss_S + 0.9*loss #+0.3*loss_2
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.idxs)/self.args.local_bs)+1, loss.item(), running_corrects/len(self.idxs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print(" ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_TS(self, net, w_glob_keys, global_round, idx):
        net.train()
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_s,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr_s,
                                         weight_decay=1e-4)

        local_eps = self.args.local_TS_ep
        epoch_loss = []
        num_updates = 0
        
        for name, param in net.named_parameters(): #####冻结backbone
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for iter in range(local_eps):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                net.zero_grad()
                log_probs,log_probs_S = net(images)
                loss = self.criterion(log_probs_S, labels)
                loss_S = self.criterion(log_probs_S, log_probs)
                #loss_2 = self.criterion(log_probs, labels)
                loss = 0.1*loss_S + 0.9*loss #+0.3*loss_2
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.idxs)/self.args.local_bs)+1, loss.item(), running_corrects/len(self.idxs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print(" ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model, idx, task):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        with torch.no_grad():
            running_corrects_test = torch.zeros(1).squeeze().to(self.device)
            labels = []
            y_pred_list = [] 

            for batch_idx, dataset in enumerate(self.testloader):
                image,label = dataset
                inputs1 = Variable(image).to(self.device)
                target = Variable(label).to(self.device)
                # Inference
                out,_ = model(inputs1)
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
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        elif task == "MH-breast-HR":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        elif task == "OCT":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        elif task == "Messidor":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
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





class LocalUpdate_personal(object):
    def __init__(self, args, dataset_train, logger, idx, idxs):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(dataset_train,idxs)
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        #'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.MSE = nn.MSELoss().to(self.device)
        self.KL = nn.KLDivLoss(reduction='mean').to(self.device)
        self.idx = idx
        self.idxs = idxs
        #print(self.args.task)

    def train_test(self, dataset_train, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(len(idxs)*0.8)]
        idxs_test = idxs[int(len(idxs)*0.8):]
        #print(len(idxs_train),len(idxs_test))
        trainloader = DataLoader(DatasetSplit_train(dataset_train,self.args.task, idxs_train),batch_size=self.args.local_bs, shuffle=True)
        #testloader = DataLoader(DatasetSplit(dataset_test,self.args.task, idx),batch_size=1, shuffle=True)
        testloader = DataLoader(DatasetSplit_train(dataset_train,self.args.task, idxs_test),batch_size=self.args.local_bs, shuffle=True)
        #print(len(trainloader),len(testloader))
        return trainloader, testloader


    def train_backbone(self, net, w_glob_keys, global_round, idx):
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
        
        for name, param in net.named_parameters(): #####冻结Student
            if name in w_glob_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for iter in range(local_eps):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                net.zero_grad()
                log_probs, log_probs_S = net(images,1)
                loss = self.criterion(log_probs, labels)
                loss_S = self.criterion(log_probs_S, labels)
                #loss_2 = self.criterion(log_probs_2, labels)
                loss = 0.1*loss_S + 0.9*loss #+0.3*loss_2
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.trainloader)), loss.item(), running_corrects/(len(self.trainloader)*self.args.local_bs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if idx==6:
           torch.save(net, "/disk1/xly/model_save_class_1220/"+str(global_round + 1)+"backbone_my_model.pth")
        print(" ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_TS(self, net, w_glob_keys, global_round, idx):
        net.train()
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_s,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr_s,
                                         weight_decay=1e-4)

        local_eps = self.args.local_TS_ep
        epoch_loss = []
        num_updates = 0
        
        for name, param in net.named_parameters(): #####冻结backbone
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for iter in range(local_eps):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                net.zero_grad()
                log_probs,log_probs_S = net(images,2)
                loss = self.criterion(log_probs_S, labels)
                #loss_S = self.KL(F.log_softmax(log_probs_S,dim=1), F.log_softmax(log_probs,dim=1))
                loss_S = self.criterion(log_probs_S, log_probs)
                #loss_2 = self.criterion(log_probs, labels)
                loss = 0.1*loss_S + 0.9*loss #+0.3*loss_2
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.trainloader)), loss.item(), running_corrects/(len(self.trainloader)*self.args.local_bs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if idx==6:
           torch.save(net, "/disk1/xly/model_save_class_1220/"+str(global_round + 1)+"TS_my_model.pth")
        print(" ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model, idx, task):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        with torch.no_grad():
            running_corrects_test = torch.zeros(1).squeeze().to(self.device)
            labels = []
            y_pred_list = [] 

            for batch_idx, dataset in enumerate(self.testloader):
                image,label = dataset
                inputs1 = Variable(image).to(self.device)
                target = Variable(label).to(self.device)
                # Inference
                out,_ = model(inputs1,1)
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
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        elif task == "MH-breast-HR":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/200,m_F1_score),end='')
        elif task == "OCT":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/1000,m_F1_score),end='')
        elif task == "Messidor":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        print("         ")



    def inference_all(self, model, idx, task):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        with torch.no_grad():
            running_corrects_test = torch.zeros(1).squeeze().to(self.device)
            running_corrects_test1 = torch.zeros(1).squeeze().to(self.device)
            labels = []
            y_pred_list = [] 
            y_pred_list_1 = [] 

            for batch_idx, dataset in enumerate(self.testloader):
                image,label = dataset
                inputs1 = Variable(image).to(self.device)
                target = Variable(label).to(self.device)
                # Inference
                out,out1 = model(inputs1,1)
                
                _ , prediction =  torch.max(out,1)
                running_corrects_test = running_corrects_test + torch.sum(prediction == target)
                labels.extend(target)
                y_pred_list.extend(prediction)

                _ , prediction1 =  torch.max(out1,1)
                running_corrects_test1 = running_corrects_test1 + torch.sum(prediction1 == target)
                y_pred_list_1.extend(prediction1)
                

            A = torch.stack(labels).reshape(-1)
            #print(len(A))
            B = torch.stack(y_pred_list).reshape(-1)
            C = torch.stack(y_pred_list_1).reshape(-1)
            m_F1_score = f1_score(A.cpu(), B.cpu(), average='macro')
            m_F1_score1 = f1_score(A.cpu(), C.cpu(), average='macro')
            
        if task == "breast":
            if self.idx == 0:
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        elif task == "MH-breast-HR":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/200,m_F1_score),end='')
        elif task == "OCT":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/1000,m_F1_score),end='')
        elif task == "Messidor":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        print("         ")

        if task == "breast":
            if self.idx == 0:
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/len(self.testloader),m_F1_score1),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/len(self.testloader),m_F1_score1),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/len(self.testloader),m_F1_score1),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/len(self.testloader),m_F1_score1),end='')
        elif task == "MH-breast-HR":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/200,m_F1_score1),end='')
        elif task == "OCT":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/1000,m_F1_score1),end='')
        elif task == "Messidor":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test1/len(self.testloader),m_F1_score1),end='')
        print("         ")


def data_read_EndoPolyp(path): 
    #print(path.shape)
    #print(path[0])
    img = Image.open(path[0])
    img.convert('RGB') 
    img = img.resize((256,256),Image.ANTIALIAS)
    img_label = Image.open(path[1]).resize((256,256),Image.ANTIALIAS)
    if "CVC-ColonDB" in path[0]:
        img_label = np.array(img_label)
        ret,label = cv2.threshold(img_label,127,1, cv2.THRESH_BINARY)
    if "ETIS-LaribPolypDB" in path[0]:
        img_label = np.array(img_label)
        ret,label = cv2.threshold(img_label,127,1, cv2.THRESH_BINARY)
    if "CVC-ClinicDB" in path[0]:
        img_label = img_label.convert('L')
        img_label = np.array(img_label)
        ret,label = cv2.threshold(img_label,127,1, cv2.THRESH_BINARY)
    if "Kvasir-SEG" in path[0]:
        img_label = img_label.convert('L')
        img_label = np.array(img_label)
        ret,label = cv2.threshold(img_label,127,1, cv2.THRESH_BINARY)
    apply_transform = transforms.Compose(
      [transforms.ToTensor()])
    #transforms.Normalize((0.4967411, 0.31040248, 0.2248057), (0.3161397, 0.22419964, 0.18269733))
       # ## [0.35381392, 0.19792648, 0.109140955], normStd = [0.2853565, 0.1724628, 0.12016233])
    img = apply_transform(img)
    # #print(img[0][1])
  
    return img, np.array(label).astype(int)



def data_read_PMR(path): 
    #print(path.shape)
    #print(path[0])
    if "Promise12" in path[0]:
        img = np.load(path[0])
        
        #img = img.resize((256,256),Image.ANTIALIAS)
        label = np.load(path[1])
        #print(img.shape)
        apply_transform = transforms.Compose(
          [transforms.ToTensor()])
        resize = transforms.Resize([384,384])
        #transforms.Normalize((0.4967411, 0.31040248, 0.2248057), (0.3161397, 0.22419964, 0.18269733))
           # ## [0.35381392, 0.19792648, 0.109140955], normStd = [0.2853565, 0.1724628, 0.12016233])
        img = apply_transform(img)
        #print(img)
        img = resize(img).float()
        label = resize(apply_transform(label)).squeeze(0).float()
        a = np.array(label).astype(int)
        #np.savetxt("./a.txt",a)

        
    
    else:
        img = np.load(path[0])
        img = img.reshape(1,384,384)
        #img = img.resize((256,256),Image.ANTIALIAS)
        label = np.load(path[1])
        # apply_transform = transforms.Compose([transforms.ToTensor()])
        # img = apply_transform(img)
  
    return img, np.array(label).astype(int)






class DatasetSplit_Seg(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, path, task):
        self.dataset = path[:]
        self.task = task
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item): 
        if self.task == "EndoPolyp":
           image, label = data_read_EndoPolyp(self.dataset[item])
        if self.task == "PMR":
           image, label = data_read_PMR(self.dataset[item])
        #print(image.dtype)
        #print(torch.tensor(image))
        return image, label

class LocalUpdatepFedCS(object):
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
        trainloader = DataLoader(DatasetSplit_Seg(dataset_train,self.args.task),batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit_Seg(dataset_test,self.args.task),batch_size=1, shuffle=True)
        return trainloader, testloader

    def train_backbone(self, net, w_glob_keys, global_round,idx):
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
        for name, param in net.named_parameters(): #####冻结Student
            if name in w_glob_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for iter in range(local_eps):


            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                out, out1 = net(images,1)
                loss = dice_loss(
                                    F.softmax(out/2, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.4*self.loss_func(out/2, labels)+ 0.1* self.loss_func(out1, labels)+0.5*loss
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                # if batch_idx % 2 == 0:
                print('\r','| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()),end='')
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if self.idx==0:
           torch.save(net, "/disk1/xly/model_save_class_1220/"+str(global_round + 1)+"backbone_my_model.pth")
        print("         ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_TS(self, net, w_glob_keys, global_round,lr=0.00001):
        net.train()
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_s,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr_s,
                                         weight_decay=1e-4)

        local_eps = self.args.local_TS_ep
        epoch_loss = []
        num_updates = 0
        
        for name, param in net.named_parameters(): #####冻结backbone
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                out, out1 = net(images,2)
                loss = dice_loss(
                                    F.softmax(out/2, dim=1).float(),
                                    F.one_hot(labels,2).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                loss = 0.1*self.loss_func(out1/2, out/2) + 0.4*self.loss_func(out1, labels)+0.5*loss
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                # if batch_idx % 2 == 0:
                print('\r','| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()),end='')
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if self.idx==0:
           torch.save(net, "/disk1/xly/model_save_class_1220/"+str(global_round + 1)+"TS_my_model.pth")
        print("         ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


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
                mask_pred,_ = model(inputs,2)             #得到前向传播的结果
                #print(out.size())
                mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #print(mask_true.size(),mask_pred.size())
                #print(dice.item())
                dice_list.append(dice.item())
                #print(dice)
  
                if self.idx == 0:
                    print('\r','CVC-ColonDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                    # torch.save(model, "/home/xly/pFred_medical_segmentation/save_model/pFLSC_person/"+str(epoch + 1)+"my_model.pth")
                if self.idx == 1:
                    print('\r','ETIS-LaribPolypDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if self.idx == 2:
                    print('\r','CVC-ClinicDB dice:{:.4f}'.format(np.mean(dice_list)),end='')
                if self.idx == 3:
                    print('\r','Kvasir-SEG dice:{:.4f}'.format(np.mean(dice_list)),end='')
            print("         ")



class DatasetSplit_TimeS(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, labels, idxs):
        self.dataset = data
        self.labels = labels
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item): 
        image = self.dataset[self.idxs[item]]
        label = self.labels[self.idxs[item]].type(torch.LongTensor)
            
        #print(label)
        #print(torch.tensor(image))
        return image, label
        
        
        
class LocalUpdate_personal_TimeS(object):
    def __init__(self, args, dataset_train, dataset_train_labels, logger, idx, idxs):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(dataset_train,dataset_train_labels, idxs)
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        #'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.MSE = nn.MSELoss().to(self.device)
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.idx = idx
        self.idxs = idxs
        #print(self.args.task)

    def train_test(self, dataset_train, dataset_train_labels, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(len(idxs)*0.8)]
        idxs_test = idxs[int(len(idxs)*0.8):]
        #print(len(idxs_train),len(idxs_test))
        trainloader = DataLoader(DatasetSplit_TimeS(dataset_train,dataset_train_labels, idxs_train),batch_size=self.args.local_bs, shuffle=True)
        #testloader = DataLoader(DatasetSplit(dataset_test,self.args.task, idx),batch_size=1, shuffle=True)
        testloader = DataLoader(DatasetSplit_TimeS(dataset_train,dataset_train_labels, idxs_test),batch_size=self.args.local_bs, shuffle=True)
        #print(len(trainloader),len(testloader))
        return trainloader, testloader


    def train_backbone(self, net, w_glob_keys, global_round, idx):
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
        
        for name, param in net.named_parameters(): #####冻结Student
            if name in w_glob_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for iter in range(local_eps):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                net.zero_grad()
                log_probs, log_probs_S = net(images,1)
                loss = self.criterion(log_probs, labels)
                loss_S = self.criterion(log_probs_S, labels)
                #loss_2 = self.criterion(log_probs_2, labels)
                loss = 0.1*loss_S + 0.9*loss #+0.3*loss_2
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.trainloader)), loss.item(), running_corrects/(len(self.trainloader)*self.args.local_bs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print(" ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_TS(self, net, w_glob_keys, global_round, idx):
        net.train()
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_s,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr_s,
                                         weight_decay=1e-4)

        local_eps = self.args.local_TS_ep
        epoch_loss = []
        num_updates = 0
        
        for name, param in net.named_parameters(): #####冻结backbone
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for iter in range(local_eps):
            batch_loss = []
            running_corrects = torch.zeros(1).squeeze().to(self.device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #print("ccccccccc",labels)
                net.zero_grad()
                log_probs,log_probs_S = net(images,2)
                loss = self.criterion(log_probs_S, labels)
                loss_S = self.KL(F.softmax(log_probs_S,dim=1), F.softmax(log_probs,dim=1))
                #loss_S = self.criterion(log_probs_S, log_probs)
                #loss_2 = self.criterion(log_probs, labels)
                loss = 0.1*loss_S + 0.9*loss #+0.3*loss_2
                loss.backward()
                optimizer.step()
                _ , prediction =  torch.max(log_probs,1)
                running_corrects = running_corrects + torch.sum(prediction == labels)
                print('\r','Idx{}, Epoch[{}/{}],Process[{}/{}],loss:{:.6f},ACC:{:.6f}'.format(idx, global_round + 1, 100, batch_idx + 1, int(len(self.trainloader)), loss.item(), running_corrects/(len(self.trainloader)*self.args.local_bs)), end='')
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print(" ")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model, idx, task):
        """ Returns the inference accuracy and loss.
        """
        
        model.eval()
        with torch.no_grad():
            running_corrects_test = torch.zeros(1).squeeze().to(self.device)
            labels = []
            y_pred_list = [] 

            for batch_idx, dataset in enumerate(self.testloader):
                image,label = dataset
                inputs1 = Variable(image).to(self.device)
                target = Variable(label).to(self.device)
                # Inference
                out,_ = model(inputs1,1)
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
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 1:
                print('\r','x2 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 2:
                print('\r','x4 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
            if self.idx == 3:
                print('\r','x8 TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/len(self.testloader),m_F1_score),end='')
        elif task == "MH-breast-HR":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/200,m_F1_score),end='')
        elif task == "OCT":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/1000,m_F1_score),end='')
        elif task == "Time-series":
                print('\r','HR TestACC:{:.4f},F1:{:.4f}'.format(running_corrects_test/2227,m_F1_score),end='')
        print("         ")