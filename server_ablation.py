import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np

from utils.aggregator import *
from utils.dispatchor import dispatch
from utils.optimizer import fedprox_optimizer
from utils.clusteror import cluster_clients
from utils.global_test import *
from utils.local_test import *
from utils.sampling import client_sampling
from utils.AnchorLoss import AnchorLoss,ContrastiveLoss
from utils.CocoLoss import CocoLoss
from utils.CKA import linear_CKA, kernel_CKA

import model
from client import client_update, client_prox_update, client_finetune, client_joint_finetune, client_fedfa, client_fedfa_cl,client_mutual_update,client_pmutual_finetune,client_finetune1,client_pmutual_finetune_c
 
from client import *

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
        self.nns = []
        self.p_nns = []
        self.cls = []
        self.cls_rand_init = []
        self.cls_ideal_init = []
        self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict =  dict((k, [0]) for k in key)
        self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict =  dict((i, []) for i in range(args.r))
        self.dataset = dataset
        self.dict_users = dict_users
        
        self.nns = [[] for i in range(args.K)]

            
        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2) 
            
        self.anchorloss_rand_init = AnchorLoss(self.args.num_classes, self.args.dims_feature, ablation=1).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss_rand_init)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls_rand_init.append(temp2) 
            
        self.anchorloss_ideal_init = AnchorLoss(self.args.num_classes, self.args.dims_feature, ablation=2).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss_ideal_init)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls_ideal_init.append(temp2) 
    
    def fedfa_without_anchor_updating(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # 在K个客户端中采样m个clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)
            
            
            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedfa_cl_without_anchor_updating(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            

            # aggregation
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    
    def fedfa_without_classifer_calibration(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # 在K个客户端中采样m个clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)
            
            
            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedfa_cl_without_classifer_calibration(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    def fedfa_with_post_classifer_calibration(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
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
            dispatch(index, self.anchorloss, self.cls)
            
            
            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedfa_cl_with_post_classifer_calibration(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    def fedfa_with_epoch_classifer_calibration(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
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
            dispatch(index, self.anchorloss, self.cls)
            
            
            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedfa_cl_with_epoch_classifer_calibration(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    def fedfa_with_pre_classifer_calibration(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
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
            dispatch(index, self.anchorloss, self.cls)
            
            
            #joint updating to obtain personalzied model based on updating global model
            self.cls, self.nns, self.loss_dict  = client_fedfa_cl_with_pre_classifer_calibration(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    def fedfa_without_anchor_specfic_initialization(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
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
            dispatch(index, self.anchorloss_rand_init, self.cls_rand_init)
            
            
            #local training
            self.cls_rand_init, self.nns, self.loss_dict  = client_fedfa_cl_without_anchor_specfic_initialization(self.args,index, 
                                                                            self.cls_rand_init,
                                                                  self.nns, self.nn, t, self.dataset,
                                                                  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss_rand_init, self.cls_rand_init, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
###
    def fedfa_with_anchor_oneround_initialization(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
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
            dispatch(index, self.anchorloss_rand_init, self.cls_rand_init)
            
            
            #local training
            self.cls_rand_init, self.nns, self.loss_dict  = client_fedfa_cl_with_anchor_oneround_initialization(self.args,index, 
                                                                            self.cls_rand_init,
                                                                  self.nns, self.nn, t, self.dataset,
                                                                  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss_rand_init, self.cls_rand_init, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
###
    def fedfa_with_anchor_ideal_initialization(self, testset, dict_users_test, CKA=False, 
                         test_global_model_accuracy = False):
        acc_list = []
        mean_CKA_dict = {"linear":[], "kernel":[]}
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
            dispatch(index, self.anchorloss_rand_init, self.cls_ideal_init)
            
            
            #local training
            self.cls_ideal_init, self.nns, self.loss_dict  = client_fedfa_cl_without_anchor_specfic_initialization(self.args,index, 
                                                                            self.cls_ideal_init,
                                                                  self.nns, self.nn, t, self.dataset,
                                                                  self.dict_users, self.loss_dict) 
            
            # aggregation
            aggregation(index, self.anchorloss_rand_init, self.cls_ideal_init, self.dict_users)
            aggregation(index, self.nn, self.nns, self.dict_users)
            
            
            if test_global_model_accuracy:
                acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                acc_list.append(acc)
                
        mean_CKA_dict = acc_list
        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict
    
    def fedbn(self, testsets, dict_test, clients_dataset_index, 
              fe_optimizer_name = "fedfa",test_global_model_accuracy = False):
    
        mean_CKA_dict = {"linear":[], "kernel":[]}
        acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
        datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

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

            # dispatch
            #self.dispatch(index)
            if exist_bn:
                for k in index:
                    self.nns[k] = copy.deepcopy(self.nn)
                    path="results/Ablation/feature skew/{}/{}/seed{}/client{}_model_{}_{}E_{}class".format(
                                                        self.args.dataset, fe_optimizer_name,
                                                        self.args.seed,k,fe_optimizer_name, self.args.E, 
                                                            self.args.split)
                    global_w = self.nn.state_dict()
                    client_w = torch.load(path)
                    for key in global_w:
                        if 'bn' not in key:
                            client_w[key] = global_w[key] 
                    self.nns[k].load_state_dict(client_w)
            else:
                dispatch(index, self.nn, self.nns)

            # # local updating
            if  fe_optimizer_name == "fedfa":
                dispatch(index, self.anchorloss, self.cls)
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.cls, self.nns, loss_dict = client_fedfa_cl(self.args, k_set, self.cls, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
                aggregation(index, self.anchorloss, self.cls, self.dict_users)
                
            elif fe_optimizer_name == "fedfa_without_anchor_updating":
                dispatch(index, self.anchorloss, self.cls)
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.cls, self.nns, loss_dict = client_fedfa_cl_without_anchor_updating(self.args, k_set, self.cls, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
                aggregation(index, self.anchorloss, self.cls, self.dict_users)
                
            elif fe_optimizer_name == "fedfa_without_classifer_calibration":
                dispatch(index, self.anchorloss, self.cls)
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.cls, self.nns, loss_dict = client_fedfa_cl_without_classifer_calibration(self.args, k_set, self.cls, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
                aggregation(index, self.anchorloss, self.cls, self.dict_users)
                
            elif fe_optimizer_name == "fedfa_without_anchor_specfic_initialization":
                dispatch(index, self.anchorloss, self.cls_rand_init)
                for k in index:
                    client_dataset_index = clients_dataset_index[k]
                    k_set = [k]
                    loss_dict = copy.deepcopy(self.loss_dict)
                    self.cls_rand_init, self.nns, loss_dict = client_fedfa_cl_without_anchor_specfic_initialization(self.args, k_set, self.cls_rand_init, self.nns, self.nn, t, self.dataset[client_dataset_index],  self.dict_users,  loss_dict)
                    self.loss_dict[k] = loss_dict[k]
                    
                aggregation(index, self.anchorloss, self.cls_rand_init, self.dict_users)
                
            index_nonselect = list(set(i for i in range(self.args.K)) - set(index))
            #print(index_nonselect)
            for j in index_nonselect:
                loss = [loss_dict[j][-1]]*self.args.E 
                self.loss_dict[j].extend(loss)   
            
 

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
                                path="results/Ablation/feature skew/{}/{}/seed{}/client{}_model_{}_{}E_{}class".format(
                                                    self.args.dataset,fe_optimizer_name, self.args.seed,
                                                        k, fe_optimizer_name,self.args.E, 
                                                self.args.split)
                                global_w = test_g_nn.state_dict()
                                client_w = torch.load(path)
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
                        print(datasets_name[testset_index], ":", mean_acc)
                        acc_list_dict[datasets_name[testset_index]].append(mean_acc)
                    del test_g_nn
                    torch.cuda.empty_cache()
                else:
                    for testset_index, testset in enumerate(testsets):
                        acc,_ = test_on_globaldataset_mixed_digit(self.args, self.nn, testset,
                                                                dict_test[datasets_name[testset_index]])
                        print(datasets_name[testset_index], ":", acc)
                        acc_list_dict[datasets_name[testset_index]].append(acc)
            
            self.nns = [[] for i in range(self.args.K)]
            for i in range(len(index)):
                torch.cuda.empty_cache()
        mean_CKA_dict = acc_list_dict

        return self.nn, self.p_nns, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict