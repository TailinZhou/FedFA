import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np



import model
from client import *
from utils.aggregator import *
from utils.dispatchor import *
from utils.optimizer import *
from utils.clusteror import *
from utils.global_test import *
from utils.local_test import *
from utils.sampling import *
from utils.AnchorLoss import *
from utils.ContrastiveLoss import *
from utils.CKA import linear_CKA, kernel_CKA


def seed_torch(seed, test = True):
    if test:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Server:

    def __init__(self, args, model, dataset, dict_users):
        seed_torch(args.seed)
        self.args = args
        self.nn = copy.deepcopy(model)
        self.nns = [[] for i in range(self.args.K)]
        self.p_nns = []
        self.cls = []
        self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict =  dict((k, [0]) for k in key)
        #self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict =  dict((i, []) for i in range(args.r))
        self.dataset = dataset
        self.dict_users = dict_users
        

        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2) 
            
        self.contrastiveloss = ContrastiveLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.contrastiveloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.contrals.append(temp2) 

    def fedavg_joint_update(self, testset, dict_users_test, iid=False, fedbn=False,similarity=False,
                            test_global_model_accuracy = False):
        if fedbn:
            acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
            datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
        else:
            acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            if fedbn:
                for i in index:
                    global_w = self.nn.state_dict()
                    if self.nns[i] == []:
                        self.nns[i] = copy.deepcopy(self.nn)
                    else:
                        client_w = self.nns[i].state_dict()
                        for key in global_w:
                            if 'bn' not in key:
                                client_w[key] = global_w[key] 
                        self.nns[i].load_state_dict(client_w)
            else:
                dispatch(index, self.nn, self.nns)
            
            # # local updating
            self.nns, self.loss_dict = client_update(self.args, index, self.nns, self.nn, t, self.dataset,  self.dict_users,  self.loss_dict)
            
            # compute feature similarity
            if similarity:
                # compute feature similarity
                if fedbn:
                    mean_feature_similarity_list = []
                    for testset_index, testset_per in enumerate(testset):

                        mean_feature_similarity0 = compute_mean_feature_similarity(self.args, index,self.nns, self.dataset,  self.dict_users, 
                                                                        testset_per,dict_users_test[datasets_name[testset_index]])
                        mean_feature_similarity_list.append(mean_feature_similarity0)
                    mean_feature_similarity = torch.mean(torch.Tensor(mean_feature_similarity_list))
                else:
                    mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)
            
            # aggregation
            if fedbn:
                aggregation(index, self.nn, self.nns, self.dict_users,fedbn=True)
            else:
                aggregation(index, self.nn, self.nns, self.dict_users)
            

            if test_global_model_accuracy:
                if fedbn:
                    for index, testset_per in enumerate(testset):
                        acc,_ = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index]])
                        acc_list_dict[datasets_name[index]].append(acc)
                        print(acc)
            
                else:
                    acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                    print(acc)
            
        if fedbn:
            mean_CKA_dict = acc_list_dict
        else:
            mean_CKA_dict = acc_list

        if iid:
            for k in range(self.args.K):
                path="results/Test/{} skew/{}/iid-fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(self.args.skew,
                                                        self.args.dataset, self.args.seed,k, 
                                                                    self.args.E, self.args.split)
                if self.nns[k]!=[]:
                    torch.save(self.nns[k].state_dict(), path)
        else:
            for k in range(self.args.K):
                path="results/Test/{} skew/{}/fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(self.args.skew,
                                                    self.args.dataset, self.args.seed,k, 
                                                                self.args.E, self.args.split)
                if self.nns[k]!=[]:
                    torch.save(self.nns[k].state_dict(), path)

        self.nns = []
        torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict


    def fedprox_joint_update(self, testset, dict_users_test, CKA=False,similarity=False,test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            dispatch(index, self.nn, self.nns)
 
            # # proximal updating
            self.nns, self.loss_dict = client_prox_update(self.args, index, self.nns, self.nn, t, self.dataset,  self.dict_users,  self.loss_dict)
 

            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)

            # aggregation
            aggregation(index, self.nn, self.nns, self.dict_users)


            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                print(acc)
                
        mean_CKA_dict = acc_list
        for k in range(self.args.K):
            path="results/Test/{} skew/{}/fedprox/seed{}/client{}_model_fedprox_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E,
                                                self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)
        self.nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict

    def feddyn(self, testset, dict_users_test, CKA=False,similarity=False,test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index
            preround_nns = [[] for i in range(self.args.K)]
            for k in index:
                if self.nns[k]==[]:
                    preround_nns[k] = copy.deepcopy(self.nn)
                else:
                    preround_nns[k] = copy.deepcopy(self.nns[k])
                
            # dispatch
            #self.dispatch(index)
            dispatch(index, self.nn, self.nns)
            
            #joint updating to obtain personalzied model based on updating global model
            self.nns, self.loss_dict  = client_feddyn(self.args, index, preround_nns, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            del preround_nns
            torch.cuda.empty_cache()
            
            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)

            # aggregation
            aggregation(index, self.nn, self.nns, self.dict_users)

            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                print(acc)
                
        mean_CKA_dict = acc_list
        for k in range(self.args.K):
            path="results/Test/{} skew/{}/feddyn/seed{}/client{}_model_feddyn_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E, 
                                                 self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)


        self.nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn,similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    

    
    def moon(self, testset, dict_users_test, CKA=False,similarity=False,test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            preround_nns = [[] for i in range(self.args.K)]
            for k in index:
                if self.nns[k]==[]:
                    preround_nns[k] = copy.deepcopy(self.nn)
                else:
                    preround_nns[k] = copy.deepcopy(self.nns[k])
            # dispatch
            #self.dispatch(index)
            dispatch(index, self.nn, self.nns)

            #joint updating to obtain personalzied model based on updating global model
            self.nns, self.loss_dict  = client_moon(self.args, index,preround_nns, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            del preround_nns
            torch.cuda.empty_cache()
            
            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)
                
            # aggregation
            aggregation(index, self.nn, self.nns, self.dict_users)

            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                print(acc)
                
        mean_CKA_dict = acc_list
        for k in range(self.args.K):
            path="results/Test/{} skew/{}/moon/seed{}/client{}_model_moon_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E, self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)


        self.nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    def fedproc(self, testset, dict_users_test, CKA=False,similarity=False,test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            dispatch(index, self.nn, self.nns)
            dispatch(index, self.contrastiveloss, self.contrals)
            
            
            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedproc(self.args,index, self.contrals, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            
            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)
            
            # aggregation
            aggregation(index, self.contrastiveloss, self.contrals, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            

            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                print(acc)
                
        mean_CKA_dict = acc_list
        
        for k in range(self.args.K):
            path="results/Test/{} skew/{}/fedproc/seed{}/client{}_model_fedproc_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E, 
                                                                            self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)
        self.nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn,similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict



    def fedfa_anchorloss(self, testset, dict_users_test, similarity=False, fedbn = False,
                         test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        if fedbn:
            acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
            datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
        else:
            acc_list = []
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            # dispatch
            if fedbn:
                for i in index:
                    global_w = self.nn.state_dict()
                    client_w = self.nns[i].state_dict()
                    for key in global_w:
                        if 'bn' not in key:
                            client_w[key] = global_w[key] 
                    self.nns[i].load_state_dict(client_w)
            else:
                dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)


            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedfa_cl(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            
            
            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users,
                                                                    testset, dict_users_test)
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)
            
            # aggregation
            if fedbn:
                aggregation(index, self.nn, self.nns, self.dict_users,fedbn=True)
            else:
                aggregation(index, self.nn, self.nns, self.dict_users)
            aggregation(index, self.anchorloss, self.cls, self.dict_users)

            if test_global_model_accuracy:
                if fedbn:
                    for index1, testset_per in enumerate(testset):
                        acc,_ = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index1]])
                        acc_list_dict[datasets_name[index1]].append(acc)
                        print(acc)
            
                else:
                    acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                    print(acc)

            
        if fedbn:
            mean_CKA_dict = acc_list_dict
        else:
            mean_CKA_dict = acc_list

        for k in range(self.args.K):
            path="results/Test/{} skew/{}/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E, self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)
        self.nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict

    def fedbn(self, testsets, dict_test, clients_dataset_index, similarity=False,
              fe_optimizer_name = "fedavg",test_global_model_accuracy = False):
    
        similarity_dict = {"feature":[], "classifier":[]}
        acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
        datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
        # for k in range(self.args.K):
        #     path="results/Test/feature skew/{}/{}/seed{}/client{}_model_{}_{}E_{}class".format(
        #                                             self.args.dataset, fe_optimizer_name,
        #                                             self.args.seed,k,fe_optimizer_name, self.args.E, 
        #                                                 self.args.split)
        #     torch.save(self.nn.state_dict(), path)
        
        if fe_optimizer_name == "feddyn" or fe_optimizer_name == "moon":
            preround_nns = [[] for i in range(self.args.K)]
            #preround_nns = copy.deepcopy(self.nns)
            
        for key in self.nn.state_dict().keys():
            if 'bn' in key:
                exist_bn = True
                break
        exist_bn = False  
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            if fe_optimizer_name == "feddyn" or fe_optimizer_name == "moon":
                preround_nns = [[] for i in range(self.args.K)]
                for k in index:
                    if self.nns[k]==[]:
                        preround_nns[k] = copy.deepcopy(self.nn)
                    else:
                        preround_nns[k] = copy.deepcopy(self.nns[k])
            # dispatch
            #self.dispatch(index)
            if exist_bn:
                for i in index:
                    global_w = self.nn.state_dict()
                    client_w = self.nns[i].state_dict()
                    for key in global_w:
                        if 'bn' not in key:
                            client_w[key] = global_w[key] 
                    self.nns[i].load_state_dict(client_w)
            else:
                dispatch(index, self.nn, self.nns)

            # # local updating
            if fe_optimizer_name == "fedavg":
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.nns, loss_dict = client_update(self.args, k_set, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
            elif fe_optimizer_name == "fedprox":
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.nns, loss_dict = client_prox_update(self.args, k_set, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
            elif fe_optimizer_name == "feddyn":

                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.nns, loss_dict = client_feddyn(self.args, k_set,preround_nns, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                del preround_nns
                torch.cuda.empty_cache()
                
            elif fe_optimizer_name == "moon":

                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.nns, loss_dict = client_moon(self.args, k_set, preround_nns, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                del preround_nns
                torch.cuda.empty_cache()
                
            elif fe_optimizer_name == "fedproc":
                dispatch(index, self.contrastiveloss, self.contrals)
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.cls, self.nns, loss_dict = client_fedproc(self.args, k_set, self.contrals, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
                aggregation(index, self.contrastiveloss, self.contrals, self.dict_users)
                    
            elif fe_optimizer_name == "fedfa":
                dispatch(index, self.anchorloss, self.cls)
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.cls, self.nns, loss_dict = client_fedfa_cl(self.args, k_set, self.cls, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
                aggregation(index, self.anchorloss, self.cls, self.dict_users)
                    
            
            index_nonselect = list(set(i for i in range(self.args.K)) - set(index))
            #print(index_nonselect)
            for j in index_nonselect:
                loss = [loss_dict[j][-1]]*self.args.E 
                self.loss_dict[j].extend(loss)   
            
            
            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity_list = []
                for testset_index, testset in enumerate(testsets):
                    
                    mean_feature_similarity0 = compute_mean_feature_similarity(self.args, index, self.nns, 
                                                                  self.dataset,  self.dict_users, testset, 
                                                                dict_test[datasets_name[testset_index]])
                    mean_feature_similarity_list.append(mean_feature_similarity0)
                mean_feature_similarity = torch.mean(torch.Tensor(mean_feature_similarity_list))
                
                # compute classifier similarity
                client_classifiers = {i:[] for i in index}
                cos_sim_matrix = torch.zeros(len(index),len(index))
                for k in index:
                    classifier_weight_update =  self.nns[k].classifier.weight.data - self.nn.classifier.weight.data
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(10,1) -self.nn.classifier.bias.data.view(10,1)
                    client_classifiers[k] = torch.cat([classifier_weight_update,
                                                       classifier_bias_update],1)
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(client_classifiers[k],client_classifiers[j])
                        #print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)
                
                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)
            
            # aggregation

            aggregation(index, self.nn, self.nns, self.dict_users, fedbn = exist_bn)

            # test on all digit datasets with a test model consisting of global model and local bn layers,
            # and average the acc for each dataset
            if test_global_model_accuracy:
                print('round', t + 1, ' test_acc:')
                if exist_bn:
                    test_g_nn = copy.deepcopy(self.nn)
                    for testset_index, testset in enumerate(testsets):
                        count = 1
                        mean_acc = 0.0
                        for k in range(self.args.K):
                            torch.cuda.empty_cache()
                            client_dataset_index = clients_dataset_index[k]
                            if testset_index == client_dataset_index:

                                global_w = test_g_nn.state_dict()
                                client_w = self.nns[k].state_dict()
                                for key in global_w:
                                    if 'bn' in key:
                                        global_w[key] = client_w[key]
                                test_g_nn.load_state_dict(global_w)
                            else:
                                continue
                            acc,_ = test_on_globaldataset_mixed_digit(self.args, test_g_nn, testset,
                                                                dict_test[datasets_name[testset_index]])
                            count += 1
                            mean_acc +=acc
                        mean_acc = mean_acc/count
                        #print(datasets_name[testset_index], ":", mean_acc)
                        acc_list_dict[datasets_name[testset_index]].append(mean_acc)
                        print(mean_acc)
                    del test_g_nn
                    torch.cuda.empty_cache()
                else:
                    for testset_index, testset in enumerate(testsets):
                        acc,_ = test_on_globaldataset_mixed_digit(self.args, self.nn, testset,
                                                                dict_test[datasets_name[testset_index]])
                        #print(datasets_name[testset_index], ":", acc)
                        acc_list_dict[datasets_name[testset_index]].append(acc)
                        print(acc)
                        
            # self.nns = [[] for i in range(self.args.K)]
            # preround_nns = [[] for i in range(self.args.K)]
            
        mean_CKA_dict = acc_list_dict
        for k in range(self.args.K):
            path="results/Test/{} skew/{}/fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, 
                                                            self.args.E, self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)
        self.nns = [[] for i in range(self.args.K)]
        if fe_optimizer_name == "feddyn" or fe_optimizer_name == "moon":
            preround_nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
def compute_mean_feature_similarity(args, index, client_models, trainset, dict_users_train, testset, dict_users_test):
    pdist = nn.PairwiseDistance(p=2)
    dict_class_verify = {i: [] for i in range(args.num_classes)}
    for i in dict_users_test:
        for c in range(args.num_classes):
            if np.array(testset.targets)[i] == c:
                dict_class_verify[c].append(i)
    #dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}
    dict_clients_features = {k: [] for k in index}
    for k in index:
        # labels = np.array(trainset.targets)[list(dict_users_train[k])]
        # labels_class = set(labels.tolist())
        #for c in labels_class:
        for c in range(args.num_classes):
            features_oneclass = verify_feature_consistency(args, client_models[k], testset,
                                                                     dict_class_verify[c])
            features_oneclass = features_oneclass.view(1,features_oneclass.size()[0],
                                                        features_oneclass.size()[1])
            if c ==0:
                dict_clients_features[k] = features_oneclass
            else:
                dict_clients_features[k] = torch.cat([dict_clients_features[k],features_oneclass])
            
    cos_sim_matrix = torch.zeros(len(index),len(index))
    for p, k in enumerate(index):
        for q, j in enumerate(index):
            for c in range(args.num_classes):
                cos_sim0 = pdist(dict_clients_features[k][c],
                                  dict_clients_features[j][c])
                # cos_sim0 = torch.cosine_similarity(dict_clients_features[k][c],
                #                                   dict_clients_features[j][c])
                # cos_sim0 = get_cos_similarity_postive_pairs(dict_clients_features[k][c],
                #                                    dict_clients_features[j][c])
                if c ==0:
                    cos_sim = cos_sim0
                else:
                    cos_sim = torch.cat([cos_sim,cos_sim0])
            cos_sim_matrix[p][q] = torch.mean(cos_sim)
    mean_feature_similarity = torch.mean(cos_sim_matrix)

    return mean_feature_similarity

def get_cos_similarity_postive_pairs(target, behaviored):
    attention_distribution_mean = []
    for j in range(target.size(0)):
        attention_distribution = []
        for i in range(behaviored.size(0)):
            attention_score = torch.cosine_similarity(target[j], behaviored[i].view(1, -1))  
            attention_distribution.append(attention_score)
        attention_distribution = torch.Tensor(attention_distribution)
        mean = torch.mean(attention_distribution)
        attention_distribution_mean.append(mean)
    attention_distribution_mean = torch.Tensor(attention_distribution_mean)
    return attention_distribution_mean
