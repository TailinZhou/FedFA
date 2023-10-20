import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils.optimizer as op


#class client:
def client_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedavg_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict

def client_update_iid_feature(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedavg_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict



def client_prox_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedprox_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict

def client_LC_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))
        client_models[k], loss = op.fedLC_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict


def client_feddyn(args, client_index, pre_client_models, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    n_clnt = args.K
    weight_list = np.asarray([len(dict_users[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} training...'.format(k))

        client_models[k],  loss = op.feddyn_optimizer(args, pre_client_models[k], client_models[k], global_model, global_round, dataset_train, dict_users[k],weight_list[k]) 
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict



def client_moon(args, client_index, preround_client_models, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_moon...'.format(k))

        client_models[k], loss = op.moon_optimizer(args, preround_client_models[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               

    return client_models, loss_dict

def client_fedproc(args, client_index, contrastiveloss_funcs, client_models, global_model, global_round, 
                   dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index:#k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        contrastiveloss_funcs[k], client_models[k], loss = op.fedproc_optimizer(args, contrastiveloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return contrastiveloss_funcs, client_models, loss_dict





def client_fedfa_cl(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict



######################ablation####################### >>>>>> fedfa_without_anchor_updating
def client_fedfa_cl_without_anchor_updating(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_without_anchor_updating(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_without_classifer_calibration
def client_fedfa_cl_without_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_without_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_with_epoch_classifer_calibration
def client_fedfa_cl_with_post_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_post_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_with_epoch_classifer_calibration
def client_fedfa_cl_with_epoch_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_epoch_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_with_epoch_classifer_calibration
def client_fedfa_cl_with_pre_classifer_calibration(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_pre_classifer_calibration(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

######################ablation####################### >>>>>> fedfa_without_anchor_specfic_initialization
def client_fedfa_cl_without_anchor_specfic_initialization(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_without_anchor_specfic_initialization(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict


######################ablation####################### >>>>>> fedfa_with_anchor_oneround_initialization
def client_fedfa_cl_with_anchor_oneround_initialization(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    for k in client_index: #k is the index of the client
        if args.verbose:
            print('client {} client_fedfa_anchorloss...'.format(k))

        anchorloss_funcs[k], client_models[k], loss = op.fedfa_with_anchor_oneround_initialization(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

