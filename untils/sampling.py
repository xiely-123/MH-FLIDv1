#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random


def dataset_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def dataset_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from breast dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = num_users, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset[:,1].astype(np.int)#.train_labels.numpy()
    #print(len(dataset[:,1].astype(np.int)))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #print(idxs_labels)
    #print(idxs_labels)
    idxs = idxs_labels[0, :]

    # divide and assign 5 shards/client
    for i in range(num_users):
        rand_set = {i}#set(np.random.choice(idx_shard, 1, replace=False))
        #rand_set = {i}#set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            idxs_shuffle = idxs[rand*num_imgs:(rand+1)*num_imgs]
            random.shuffle(idxs_shuffle)
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs_shuffle), axis=0)
    #print(int(dict_users[1]))
    return dict_users

def dataset_noniid_personal(dataset, num_users):
    """
    Sample non-I.I.D client data from breast dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 32, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset[:,1].astype(np.int)#.train_labels.numpy()
    #print(len(dataset[:,1].astype(np.int)))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #print(idxs_labels)
    #print(idxs_labels)
    idxs = idxs_labels[0, :]

    # divide and assign 5 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            #print(idxs[rand*num_imgs:(rand+1)*num_imgs])
            idxs_shuffle = idxs[rand*num_imgs:(rand+1)*num_imgs]
            #random.shuffle(idxs_shuffle)
            #print(idxs_shuffle)
            #idxs[rand*num_imgs:(rand+1)*num_imgs]
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs_shuffle), axis=0)
            #print("aaaaaaaaaaaaa",dict_users[i])
            random.shuffle(dict_users[i])####随机，保证test与train分布一致
            #print("bbbbbbbbbbbbb",dict_users[i])
            #idxs_shuffle
    return dict_users


def dataset_noniid_personal_OCT(dataset, num_users):
    """
    Sample non-I.I.D client data from breast dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 40, 1000   ####40 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset[:,1].astype(np.int)#.train_labels.numpy()
    #print(len(dataset[:,1].astype(np.int)))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #print(idxs_labels)
    #print(idxs_labels)
    idxs = idxs_labels[0, :]

    # divide and assign 5 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 5, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            #print(idxs[rand*num_imgs:(rand+1)*num_imgs])
            idxs_shuffle = idxs[rand*num_imgs:(rand+1)*num_imgs]
            #random.shuffle(idxs_shuffle)
            #print(idxs_shuffle)
            #idxs[rand*num_imgs:(rand+1)*num_imgs]
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs_shuffle), axis=0)
            #print("aaaaaaaaaaaaa",dict_users[i])
            random.shuffle(dict_users[i])####随机，保证test与train分布一致
            #print("bbbbbbbbbbbbb",dict_users[i])
            #idxs_shuffle
    return dict_users



def dataset_noniid_personal_timeseries(dataset, dataset_labels, num_users):
    """
    Sample non-I.I.D client data from breast dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 12, 2783   ####40 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset_labels).astype(np.int)#.train_labels.numpy()
    #print(len(dataset[:,1].astype(np.int)))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #print(idxs_labels)
    #print(idxs_labels)
    idxs = idxs_labels[0, :]

    # divide and assign 5 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            #print(idxs[rand*num_imgs:(rand+1)*num_imgs])
            idxs_shuffle = idxs[rand*num_imgs:(rand+1)*num_imgs]
            #random.shuffle(idxs_shuffle)
            #print(idxs_shuffle)
            #idxs[rand*num_imgs:(rand+1)*num_imgs]
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs_shuffle), axis=0)
            #print("aaaaaaaaaaaaa",dict_users[i])
            random.shuffle(dict_users[i])####随机，保证test与train分布一致
            #print("bbbbbbbbbbbbb",dict_users[i])
            #idxs_shuffle
    return dict_users






# if __name__ == '__main__':
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   # transform=transforms.Compose([
                                       # transforms.ToTensor(),
                                       # transforms.Normalize((0.1307,),
                                                            # (0.3081,))
                                   # ]))
    # num = 100
    # d = mnist_noniid(dataset_train, num)
