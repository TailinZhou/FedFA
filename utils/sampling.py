#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random
from torchvision import datasets, transforms

def client_sampling(args):
    
    m = np.max([int(args.C * args.K), 1]) 
    index = random.sample(range(0, args.K), m)   
    
    return index

def trainset_sampling_label(args, trainset, sample_rate, rare_class_nums, noniid_labeldir_part):
    dict_users_train = {i: [] for i in range(args.K)}
    for i in range(args.K):

        full_dict_user_train = noniid_labeldir_part.client_dict[i]

        labels = np.array(trainset.targets)[full_dict_user_train].tolist()
        labels_class = list(set(labels))
        
        idxs_labels = np.vstack((full_dict_user_train, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

        dict_class = {i: [] for i in range(args.num_classes)}
        index_next_class = 0
        b = []
        d = list(np.random.choice(labels_class, rare_class_nums, replace=False))
        #取得各个类别所对应的索引
        for k in range(args.num_classes):
            len_class = list(idxs_labels[1,:]).count(k)

            index_curent_class = index_next_class
            index_next_class = index_curent_class + len_class
            
 
            if len_class != 0:
                
                if k in d:
                     
                    sample_num = int(sample_rate*sample_rate*len_class)
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))
                    b.extend(a)

                else:
                    sample_num = int(sample_rate*len_class)
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))

                    b.extend(a)
        dict_users_train[i] = set(b)
            
    return dict_users_train


def trainset_sampling_label_uniform(args, trainset, sample_number, rare_class_nums, noniid_labeldir_part):
    dict_users_train = {i: [] for i in range(args.K)}
    for i in range(args.K):

        full_dict_user_train = noniid_labeldir_part.client_dict[i]

        labels = np.array(trainset.targets)[full_dict_user_train].tolist()
        labels_class = list(set(labels))

        idxs_labels = np.vstack((full_dict_user_train, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

        dict_class = {i: [] for i in range(args.num_classes)}
        index_next_class = 0
        b = []
        d = list(np.random.choice(labels_class, rare_class_nums, replace=False))
        #取得各个类别所对应的索引
        for k in range(args.num_classes):
            len_class = list(idxs_labels[1,:]).count(k)

            index_curent_class = index_next_class
            index_next_class = index_curent_class + len_class
            
 
            if len_class != 0:
                
                if k in d:
                     
                    sample_num = sample_number
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))
                    b.extend(a)

                else:
                    sample_num = sample_number
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))

                    b.extend(a)
        dict_users_train[i] = set(b)
            
    return dict_users_train


def trainset_sampling(args, trainset, sample_rate, noniid_labeldir_part):
    dict_users_train = {i: [] for i in range(args.K)}
    for i in range(args.K):

        full_dict_user_train = noniid_labeldir_part.client_dict[i]

        labels = np.array(trainset.targets)[full_dict_user_train]
        idxs_labels = np.vstack((full_dict_user_train, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

        dict_class = {i: [] for i in range(args.num_classes)}
        index_next_class = 0
        b = []

        for k in range(args.num_classes):
            len_class = list(idxs_labels[1,:]).count(k)

            index_curent_class = index_next_class
            index_next_class = index_curent_class + len_class

            if len_class == 0:
                continue
            if len_class <= 240:

                dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                b.extend(dict_class[k])

            else:
                sample_num = int(sample_rate*len_class)
                dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                a = list(np.random.choice(dict_class[k], sample_num, replace=False))
                
                b.extend(a)
        dict_users_train[i] = set(b)
            
    return dict_users_train

def testset_sampling(args, testset, number_perclass, trainclass_df):
    
    idxs = np.arange(len(testset))
    labels = np.array(testset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    dict_users_test = {i: [] for i in range(args.K)}
    dict_class = {i: [] for i in range(args.num_classes)}
    index_next_class = 0

    for k in range(args.num_classes):
        len_class = list(idxs_labels[1,:]).count(k)

        index_curent_class = index_next_class
        index_next_class = index_curent_class + len_class

        dict_class[k] = list(idxs_labels[0,index_curent_class:index_next_class])

    for i in range(args.K):
        b=  list()
        for k in range(args.num_classes): 
            if trainclass_df.iloc[i][k] == 0:
                continue
            
            a = list(np.random.choice(dict_class[k], number_perclass, replace=False))
 
            b.extend(a)
   
            dict_users_test[i] = set(b)
            
    return dict_users_test

def testset_sampling1(args, testset, number_perclass, dict_clients_labelset):
    idxs = np.arange(len(testset))
    labels = np.array(testset.targets)


    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    dict_users_test = {i: [] for i in range(args.K)}
    dict_class = {i: [] for i in range(args.num_classes)}
    index_next_class = 0
    
    for k in range(args.num_classes):
        len_class = list(idxs_labels[1,:]).count(k)

        index_curent_class = index_next_class
        index_next_class = index_curent_class + len_class

        dict_class[k] = list(idxs_labels[0,index_curent_class:index_next_class])
    for i in range(args.K):
        b=  list()
        for k in range(args.num_classes): 
            if k not in dict_clients_labelset[i]:
                continue
            
            a = list(np.random.choice(dict_class[k], number_perclass, replace=False))
            b.extend(a)
            dict_users_test[i] = set(b)
            
    return dict_users_test

def trainset_sampling_mixed_digit(args, clients_index, trainset, sample_rate, rare_class_nums, labeldir_part):
    dict_users_train = {i: {} for i in clients_index}
    for j in clients_index:
        
        full_dict_user_train = labeldir_part.client_dict[clients_index.index(j)]

        labels = np.array(trainset.targets)[full_dict_user_train].tolist()
        labels_class = list(set(labels))
        idxs_labels = np.vstack((full_dict_user_train, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

        dict_class = {i: [] for i in range(args.num_classes)}
        index_next_class = 0
        b = []
        d = list(np.random.choice(labels_class, rare_class_nums, replace=False))
        
        for k in range(args.num_classes):
            len_class = list(idxs_labels[1,:]).count(k)

            index_curent_class = index_next_class
            index_next_class = index_curent_class + len_class
            
 
            if len_class != 0:
                
                if k in d:
                     
                    sample_num = int(sample_rate*sample_rate*len_class)
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))
                    b.extend(a)

                else:
                    sample_num = int(sample_rate*len_class)
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))

                    b.extend(a)
        dict_users_train[j] = set(b)
            
    return dict_users_train


def testset_sampling_mixed_digit(args, testset, number_perclass):
    np.random.seed(args.seed)
    idxs = np.arange(len(testset))
    labels = np.array(testset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

    dict_users_test = {i: [] for i in range(args.K)}
    dict_class = {i: [] for i in range(args.num_classes)}
    index_next_class = 0

    for k in range(args.num_classes):
        len_class = list(idxs_labels[1,:]).count(k)

        index_curent_class = index_next_class
        index_next_class = index_curent_class + len_class

        dict_class[k] = list(idxs_labels[0,index_curent_class:index_next_class])
    b=  list()
    for k in range(args.num_classes): 
        a = list(np.random.choice(dict_class[k], number_perclass, replace=False))
        b.extend(a)

    dict_test = set(b)

    return dict_test

def trainset_sampling_label_femnist(args, trainset, sample_rate, rare_class_nums, noniid_labeldir_part):
    dict_users_train = {i: [] for i in range(args.K)}
    for i in range(args.K):

        full_dict_user_train = noniid_labeldir_part[i]

        labels = np.array(trainset.targets)[full_dict_user_train].tolist()
        labels_class = list(set(labels))
        idxs_labels = np.vstack((full_dict_user_train, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

        dict_class = {i: [] for i in range(args.num_classes)}
        index_next_class = 0
        b = []
        d = list(np.random.choice(labels_class, rare_class_nums, replace=False))
        for k in range(args.num_classes):
            len_class = list(idxs_labels[1,:]).count(k)

            index_curent_class = index_next_class
            index_next_class = index_curent_class + len_class
            
 
            if len_class != 0:
                
                if k in d:
                     
                    sample_num = int(sample_rate*sample_rate*len_class)
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))
                    b.extend(a)

                else:
                    sample_num = int(sample_rate*len_class)
                    dict_class[k] = list(idxs_labels[0, index_curent_class:index_next_class])
                    a = list(np.random.choice(dict_class[k], sample_num, replace=False))

                    b.extend(a)
        dict_users_train[i] = set(b)
            
    return dict_users_train


def testset_sampling_femnist(args, testset, number_perclass, clients_labeset):
    idxs = np.arange(len(testset))
    labels = np.array(testset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    dict_users_test = {i: [] for i in range(args.K)}
    dict_class = {i: [] for i in range(args.num_classes)}
    index_next_class = 0

    for k in range(args.num_classes):
        len_class = list(idxs_labels[1,:]).count(k)

        index_curent_class = index_next_class
        index_next_class = index_curent_class + len_class

        dict_class[k] = list(idxs_labels[0,index_curent_class:index_next_class])

    for i in range(args.K):
        b=  list()
        for k in range(args.num_classes): 
            if k not in clients_labeset[i]:
                continue
                
            a = list(np.random.choice(dict_class[k], number_perclass, replace=False))
            b.extend(a)
            dict_users_test[i] = set(b)
            
    return dict_users_test

 