import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np
from utils.global_test import *
from utils.sampling import *

def compute_histogram(args, client_models, trainset, dict_users_train, testset, dict_users_test):

    dict_class_verify = {i: [] for i in range(args.num_classes)}
    for i in dict_users_test:
        for c in range(args.num_classes):
            if np.array(testset.targets)[i] == c:
                dict_class_verify[c].append(i)

    dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}

    for k in range(args.K):
        labels = np.array(trainset.targets)[list(dict_users_train[k])]
        labels_class = set(labels.tolist())
        for c in labels_class:
            dict_clients_features[k][c] = verify_feature_consistency(args, client_models[k], testset,
                                                                     dict_class_verify[c], numpy=False)
            
    dict_clients_labelset = {k: [] for k in range(args.K)}
    for k in range(args.K):
        labels = np.array(trainset.targets)[list(dict_users_train[k])]
        labels_class = set(labels.tolist())
        dict_clients_labelset[k] = labels_class
        
    dict_class_clientset = {c: [] for c in range(args.num_classes)}
    for c in range(args.num_classes):
        for k in range(args.K):
            if c in dict_clients_labelset[k]:
                dict_class_clientset[c].append(k)

    postive_pairs_similarity_list = []
    negative_pairs_similarity_list = []
    
    #compute the similarity of postive_pairs
    number_postive_pairs = 0
    
    for c in range(args.num_classes):
        for index, k in enumerate(dict_class_clientset[c]):
            X = dict_clients_features[k][c]
            for q in range(len(X)):
                x = X[q]
                for j in dict_class_clientset[c][index+1:]:
                    Y = dict_clients_features[j][c]
                    for p in range(len(Y)):
                        y = Y[p]
                        postive_pair_similarity = torch.cosine_similarity(x, y, dim=-1, eps=1e-08)
                        postive_pairs_similarity_list.append(postive_pair_similarity)
                        number_postive_pairs +=1

    
    
    #compute the similarity of negative_pairs
    number_negative_pairs = 0
    class_list = range(args.num_classes)
    for k in range(args.K):
        for c in dict_clients_labelset[k]:
                X = dict_clients_features[k][c]
                for q in range(len(X)):
                    x = X[q]
                    for k_n in range(args.K)[k+1:]:
                        for c_n in dict_clients_labelset[k_n]: 
                            if c_n == c:
                                continue
                            Y = dict_clients_features[k_n][c_n]
                            for p in range(len(Y)):
                                y = Y[p]
                                negative_pair_similarity = torch.cosine_similarity(x, y, dim=-1, eps=1e-08)
                                negative_pairs_similarity_list.append(negative_pair_similarity)
                                number_negative_pairs +=1

    postive_pairs_similarity_list = torch.stack(postive_pairs_similarity_list).cpu().tolist()
    negative_pairs_similarity_list = torch.stack(negative_pairs_similarity_list).cpu().tolist()
    return postive_pairs_similarity_list, negative_pairs_similarity_list


def compute_histogram_mixed_digit(args, client_models, 
                                  trainsets, dict_users_train, 
                                  testsets, dict_test, 
                                  datasets_name, clients_dataset_index, number_perclass=10):

    dict_datasets_varify = {'MNIST':{i: [] for i in range(args.num_classes)}, 
                        'SVHN':{i: [] for i in range(args.num_classes)}, 
                        'USPS':{i: [] for i in range(args.num_classes)}, 
                        'SynthDigits':{i: [] for i in range(args.num_classes)}, 
                        'MNIST-M':{i: [] for i in range(args.num_classes)}}

    dict_varify = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
    for index, testset in enumerate(testsets):
        dict_varify[datasets_name[index]] = testset_sampling_mixed_digit(args, testset, number_perclass)

    for index, testset in enumerate(testsets):
        for i in dict_varify[datasets_name[index]]: 
            for c in range(args.num_classes):
                if np.array(testset.targets)[i] == c: 
                    dict_datasets_varify[datasets_name[index]][c].append(i)
                    

    dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}

    for i in range(args.K):
        labels = np.array(trainsets[clients_dataset_index[i]].targets)[list(dict_users_train[i])]
        labels_class = list(set(labels.tolist()))
        for c in labels_class:
            index = clients_dataset_index[i]
            dict_clients_features[i][c] = verify_feature_consistency(args, 
                                                               client_models[i], 
                                                               testsets[clients_dataset_index[i]],
                                                               dict_datasets_varify[datasets_name[index]][c],
                                                               numpy=False)
            
    dict_clients_labelset = {k: [] for k in range(args.K)}
    for k in range(args.K):
        labels = np.array(trainsets[clients_dataset_index[i]].targets)[list(dict_users_train[k])]
        labels_class = set(labels.tolist())
        dict_clients_labelset[k] = labels_class
        
    dict_class_clientset = {c: [] for c in range(args.num_classes)}
    for c in range(args.num_classes):
        for k in range(args.K):
            if c in dict_clients_labelset[k]:
                dict_class_clientset[c].append(k)

    postive_pairs_similarity_list = []
    negative_pairs_similarity_list = []
    
    #compute the similarity of postive_pairs
    number_postive_pairs = 0
    
    for c in range(args.num_classes):
        for index, k in enumerate(dict_class_clientset[c]):
            X = dict_clients_features[k][c]
            for q in range(len(X)):
                x = X[q]
                for j in dict_class_clientset[c][index+1:]:
                    Y = dict_clients_features[j][c]
                    for p in range(len(Y)):
                        y = Y[p]
                        postive_pair_similarity = torch.cosine_similarity(x, y, dim=-1, eps=1e-08)
                        postive_pairs_similarity_list.append(postive_pair_similarity)
                        number_postive_pairs +=1

    
    
    #compute the similarity of negative_pairs
    number_negative_pairs = 0
    class_list = range(args.num_classes)
    for k in range(args.K):
        for c in dict_clients_labelset[k]:
                X = dict_clients_features[k][c]
                for q in range(len(X)):
                    x = X[q]
                    for k_n in range(args.K)[k+1:]:
                        for c_n in dict_clients_labelset[k_n]: 
                            if c_n == c:
                                continue
                            Y = dict_clients_features[k_n][c_n]
                            for p in range(len(Y)):
                                y = Y[p]
                                negative_pair_similarity = torch.cosine_similarity(x, y, dim=-1, eps=1e-08)
                                negative_pairs_similarity_list.append(negative_pair_similarity)
                                number_negative_pairs +=1

    postive_pairs_similarity_list = torch.stack(postive_pairs_similarity_list).cpu().tolist()
    negative_pairs_similarity_list = torch.stack(negative_pairs_similarity_list).cpu().tolist()
    return postive_pairs_similarity_list, negative_pairs_similarity_list