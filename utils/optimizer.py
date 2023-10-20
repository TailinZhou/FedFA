import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): 
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def seed_torch(seed, test = False):
    if test:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
def fedavg_optimizer(args, client_model, global_model, global_round, dataset_train, dict_user):
    
    #if set(dataset_train.targets[list(dict_user)].tolist()) != set(range(10)):
    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, weight_decay=0.0001, momentum=args.momentum)#
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)
 
    epoch_loss = []
    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            imgs = imgs.to(args.device)
            # imgs.requires_grad_(True)
            
            #print(imgs.size())
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            optimizer.zero_grad()
            

            loss = loss_function(y_preds, labels) 
            loss.backward()

            optimizer.step()
            
            if args.verbose and batch_idx % 6 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())


        
        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return client_model, epoch_loss


def fedprox_optimizer(args, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    #print(len(Dtr))
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  weight_decay=0.001,momentum=args.momentum)
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    #loss_function = nn.MSELoss().to(args.device)
    epoch_loss = []
    epoch_grad = []
    epoch_proximal_loss = []
    for epoch in range(args.E):
        batch_loss = []
        batch_grad = []
        client_model.train()
        proximal_loss_list = []
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            imgs = imgs.to(args.device)
            # imgs.requires_grad_(True)
            
            #print(imgs.size())
            labels = labels.type(torch.LongTensor).to(args.device)
            _, y_preds = client_model(imgs)
            optimizer.zero_grad()
            
            # compute proximal_term
            proximal_term = 0.0
            for w, w_t in zip(client_model.parameters(), global_model.parameters()):
                proximal_term += ((w - w_t).norm(2) **2)
            proximal_loss = (args.mu / 2) * proximal_term
            loss = loss_function(y_preds, labels) + proximal_loss
            
            loss.backward()

            optimizer.step()
            
            if args.verbose and batch_idx % 6 == 0:
                print('proximal_loss: {}'.format(proximal_loss))
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))


    return client_model, epoch_loss



def feddyn_optimizer(args, pre_client_model, client_model, global_model, global_round, dataset_train, dict_user,client_data_weight):

    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  weight_decay=0.001,momentum=args.momentum)
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)

    epoch_loss = []
    for epoch in range(args.E):
        batch_loss = []
        batch_grad = []
        client_model.train()
        proximal_loss_list = []
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            imgs = imgs.to(args.device)
            # imgs.requires_grad_(True)
            
            labels = labels.type(torch.LongTensor).to(args.device)
            _, y_preds = client_model(imgs)
            optimizer.zero_grad()
            
            # compute total loss
            loss = loss_function(y_preds, labels)
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in client_model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                    
            avg_mdl_param = None
            for param in global_model.parameters():
                if not isinstance(avg_mdl_param, torch.Tensor):
                # Initially nothing to concatenate
                    avg_mdl_param = param.reshape(-1)
                else:
                    avg_mdl_param = torch.cat((avg_mdl_param, param.reshape(-1)), 0)
                    
            local_grad_vector = None
            for param in pre_client_model.parameters():
                if not isinstance(local_grad_vector, torch.Tensor):
                # Initially nothing to concatenate
                    local_grad_vector = param.reshape(-1)
                else:
                    local_grad_vector = torch.cat((local_grad_vector, param.reshape(-1)), 0)
            
            alph = args.alph/client_data_weight
            # if args.weight_decay != 0:
            #     alph = alph * pow(args.weight_decay, global_round)
            loss_algo = alph * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            
            loss = loss + loss_algo

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=client_model.parameters(), max_norm=10) # Clip gradients
            optimizer.step()
            
            # del local_par_list, avg_mdl_param, local_grad_vector
            # torch.cuda.empty_cache()
            
            if args.verbose and batch_idx % 6 == 0:
              
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        

    return client_model, epoch_loss
    



def moon_optimizer(args, preround_client_model, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)


    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  weight_decay=0.001,momentum=args.momentum) #, weight_decay=0.0005
 
    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    epoch_loss = []
 
    for epoch in range(args.E):
        batch_loss = []
        proximal_loss_list = []
        client_model.train()
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            gfeatures, _ = global_model(imgs)
            prefeatures, _ = preround_client_model(imgs)
                                                        
            gfeatures.detach()
            prefeatures.detach()
            optimizer.zero_grad()


            # compute moon loss
            moon_similarity1 = torch.zeros(1).to(args.device)
            moon_similarity2 = torch.zeros(1).to(args.device)
            for i in range(len(imgs)):
                moon_similarity1 += torch.cosine_similarity(features[i], gfeatures[i],dim=-1, eps=1e-08)/len(imgs)
                moon_similarity2 += torch.cosine_similarity(features[i], prefeatures[i],dim=-1, eps=1e-08)/len(imgs)
                
            moon_loss = torch.exp(moon_similarity1/args.tau)/(torch.exp(moon_similarity1/args.tau) + torch.exp(moon_similarity2/args.tau))
            moon_loss = -torch.log(moon_loss)

            mu = args.mu1
            # compute total loss
            loss = loss_function(y_preds, labels) + mu*moon_loss

            loss.backward()
            optimizer.step()

            if args.verbose and batch_idx % 6 == 0:
                print('moon_loss: {}'.format(mu*moon_loss.item()))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())
        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))


    return client_model, epoch_loss


def fedproc_optimizer(args, contrastiveloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set
 
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters())
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  momentum=args.momentum,weight_decay=0.001) 
 
    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    contrastiveloss = contrastiveloss_func.to(args.device)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)

            ys = labels.float()
            loss_contrastive = contrastiveloss(features, ys)
            alpha = 1-(global_round+1)/args.r
            #loss = loss_function(y_preds, labels)  +     loss_contrastive 
            loss = (1-alpha)*loss_function(y_preds, labels)  +   alpha*loss_contrastive 
            # if loss_contrastive> 1000: 
            #     print(loss_function(y_preds, labels), loss_contrastive )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
#     update prototype parameter in contrastiveloss with the whole trainset
#     Dte = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.TB, shuffle=False)
#     epoch_mean_anchor = copy.deepcopy(contrastiveloss.anchor.data)
#     for batch_idx, (imgs, labels) in enumerate(Dte):
#         with torch.no_grad():
#             imgs = imgs.to(args.device)
#             labels = labels.type(torch.LongTensor).to(args.device)
#             updated_features, _ = client_model(imgs) 

#             for i in set(labels.tolist()):
#                 epoch_mean_anchor[i] = torch.mean(updated_features[labels==i],dim=0)
#     #contrastiveloss.anchor.data =   0.5*epoch_mean_anchor + 0.5*contrastiveloss.anchor.data
#     contrastiveloss.anchor.data =   epoch_mean_anchor  


    return contrastiveloss, client_model, epoch_loss



def fedfa_cl_optimizer(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())
 
    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    # for i in label_set:
    #     epoch_mean_anchor[i] = torch.zeros_like(epoch_mean_anchor[i])
    #anchorloss_opt = torch.optim.SGD(anchorloss.parameters(),lr=0.001)#, momentum=0.99
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            

            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            # print(loss_anchor)

            loss = loss_function(y_preds, labels)  + loss_anchor 

            #anchorloss_opt.zero_grad()
            optimizer.zero_grad()
            #optimizer_c.zero_grad()

            loss.backward()
            #anchorloss_opt.step()
            optimizer.step()
            #optimizer_c.step()
            
                    
            C = torch.arange(0,args.num_classes).to(args.device)
            x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
            # miss_label_set = list(set(range(0,args.num_classes))-label_set)
            # C = C[miss_label_set]
            # x_c = x_c[miss_label_set]

            y_c = client_model.classifier(x_c)
            loss_c = loss_function(y_c, C)
            
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            
            # #memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)
#                 batch_mean_anchor[i] = torch.mean(updated_features[labels==i],dim=0)

#                 #compute epoch mean anchor according to batch mean anchor
#                 lambda_momentum = 0.99  
#                 epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]
                
            #anchorloss.anchor.data = epoch_mean_anchor

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())


        # if epoch == 0:
        #     for i in label_set:
        #         #compute batch mean anchor according to batch label
        #         batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)
        #         epoch_mean_anchor[i] = batch_mean_anchor[i]
        # else:
        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)

            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]
        
        #anchorloss.anchor.data = epoch_mean_anchor

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    anchorloss.anchor.data =  epoch_mean_anchor



    return anchorloss, client_model, epoch_loss

######################ablation####################### >>>>>> fedfa_without_anchor_updating
def fedfa_without_anchor_updating(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())
 
    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            loss = loss_function(y_preds, labels)  +  loss_anchor 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            C = torch.arange(0,args.num_classes).to(args.device)
            x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)

            y_c = client_model.classifier(x_c)
            loss_c = loss_function(y_c, C)
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return anchorloss, client_model, epoch_loss


######################ablation####################### >>>>>> fedfa_without_classifer_calibration
def fedfa_without_classifer_calibration(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #

    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            loss = loss_function(y_preds, labels)  +  loss_anchor 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # #memorize batch feature anchor 
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)
            
            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    anchorloss.anchor.data =  epoch_mean_anchor
    
    return anchorloss, client_model, epoch_loss


def fedfa_with_pre_classifer_calibration(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())
        
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            C = torch.arange(0,args.num_classes).to(args.device)
            x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
            y_c = client_model.classifier(x_c)
            loss_c = loss_function(y_c, C)
            
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
            
            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            loss = loss_function(y_preds, labels)  +  loss_anchor 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # #memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)

            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    anchorloss.anchor.data =  epoch_mean_anchor

    return anchorloss, client_model, epoch_loss


def fedfa_with_post_classifer_calibration(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())
        
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            loss = loss_function(y_preds, labels) # +  loss_anchor 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # #memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)

            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    anchorloss.anchor.data =  epoch_mean_anchor

    return anchorloss, client_model, epoch_loss

######################ablation####################### >>>>>> fedfa_without_classifer_calibration
def fedfa_with_epoch_classifer_calibration(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())

    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)
            
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            loss = loss_function(y_preds, labels)  +  loss_anchor 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # #memorize batch feature anchor 
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)
            
            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]
            
        #calibrate classifier in each epoch
        C = torch.arange(0,args.num_classes).to(args.device)
        x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
        y_c = client_model.classifier(x_c)
        loss_c = loss_function(y_c, C)

        optimizer_c.zero_grad()
        loss_c.backward()
        optimizer_c.step()
        
        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    anchorloss.anchor.data =  epoch_mean_anchor
    
    return anchorloss, client_model, epoch_loss

######################ablation####################### >>>>>> fedfa_without_anchor_specfic_initialization
def fedfa_without_anchor_specfic_initialization(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())
 
    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)

            features, y_preds = client_model(imgs)
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            loss = loss_function(y_preds, labels)  +  loss_anchor 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            C = torch.arange(0,args.num_classes).to(args.device)
            x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
            y_c = client_model.classifier(x_c)
            loss_c = loss_function(y_c, C)
            
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            
            # #memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)

            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    anchorloss.anchor.data =  epoch_mean_anchor

    return anchorloss, client_model, epoch_loss




######################ablation####################### >>>>>> fedfa_with_anchor_oneround_initialization
def fedfa_with_anchor_oneround_initialization(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user):

    seed_torch(seed=args.seed)
    
    Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
        optimizer_c = torch.optim.Adam(client_model.classifier.parameters())
 
    
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []

    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)

            features, y_preds = client_model(imgs)
            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            if global_round == 0:
                loss = loss_function(y_preds, labels)  
            else:
                loss = loss_function(y_preds, labels)  +  loss_anchor 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            C = torch.arange(0,args.num_classes).to(args.device)
            x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
            y_c = client_model.classifier(x_c)
            loss_c = loss_function(y_c, C)
            
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            
            # #memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)

            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)

            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor #pow(2, -(epoch+1))
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    anchorloss.anchor.data =  epoch_mean_anchor

    return anchorloss, client_model, epoch_loss
