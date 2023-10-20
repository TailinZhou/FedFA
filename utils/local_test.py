#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import torch
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

def test_on_localdataset(args, client_models, dataset_test,dict_user_test):
    
    key = [i for i in range(args.K)]
    localtest_loss_dict =  dict((k, []) for k in key)
    accuracy_dict = dict((k, []) for k in key)
    
    for i in range(args.K):
        client_models[i].eval()
        # testing
        test_loss = 0.
        correct = 0.
        Dte = DataLoader(DatasetSplit(dataset_test,dict_user_test[i]), batch_size=args.TB, shuffle=True)

        l = len(Dte)
        pred = []
        y = []
        for batch_idx, (imgs, labels) in enumerate(Dte):
            with torch.no_grad():
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                _, log_probs = client_models[i](imgs)

                # sum up batch loss
                test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(Dte.dataset)
        accuracy = 100.00 * correct / len(Dte.dataset)
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(Dte.dataset), accuracy))
            
        localtest_loss_dict[i].append(test_loss)
        accuracy_dict[i].append(accuracy)
    return accuracy_dict, localtest_loss_dict     

